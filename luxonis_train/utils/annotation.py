"""Turns the predictions of a model into dataset annotations."""

from pathlib import Path
from typing import TypedDict

import cv2
import numpy as np
from luxonis_ml.data import DatasetIterator
from torch import Tensor

import luxonis_train as lxt
from luxonis_train.config.config import PreprocessingConfig
from luxonis_train.tasks import Tasks
from luxonis_train.typing import Packet

from .segmentation import seg_output_to_bool
from .spatial_transforms import (
    transform_boxes,
    transform_keypoints,
    transform_masks,
)

ALLOWED_ANNOTATE_LABELS = {
    (label if isinstance(label, str) else label.name)
    for task in (
        Tasks.BOUNDINGBOX,
        Tasks.INSTANCE_KEYPOINTS,
        Tasks.KEYPOINTS,
        Tasks.INSTANCE_SEGMENTATION,
        Tasks.SEGMENTATION,
        Tasks.CLASSIFICATION,
        Tasks.OCR,
    )
    for label in task.required_labels
}


class _Transformed(TypedDict, total=False):
    norm_boxes: np.ndarray
    norm_kpts: np.ndarray
    norm_masks: np.ndarray
    pred_classes: np.ndarray
    pred_text: list[tuple[str, float]]


def default_annotate(
    head: "lxt.nodes.BaseHead",
    head_output: Packet[Tensor],
    image_paths: list[Path],
    config_preprocessing: PreprocessingConfig,
) -> DatasetIterator:
    """Convert head output to `LuxonisDataset
    <luxonis_ml.data.datasets.LuxonisDataset>` annotations.

    Args:
        head (`luxonis_train.nodes.BaseHead`): Head from which to extract annotations.
        head_output (``Packet[Tensor]``): Output from the head containing
            predictions.
        image_paths (``list[Path]``): Paths to the images corresponding to the
            head output.
        config_preprocessing (PreprocessingConfig): Preprocessing configuration
            containing image size and aspect ratio settings.

    Yields:
        `DatasetIterator <luxonis_ml.data.datasets.DatasetIterator>`: Annotation records in a format suitable for
        `LuxonisDataset <luxonis_ml.data.datasets.LuxonisDataset>`.

    Raises:
        ValueError: If the task is unsupported or OCR output is missing a
            decoder.
        FileNotFoundError: If an input image cannot be read.

    """
    train_size = config_preprocessing.train_image_size
    keep_aspect_ratio = config_preprocessing.keep_aspect_ratio
    batch_size = len(image_paths)
    required_labels = {
        task if isinstance(task, str) else task.name
        for task in head.task.required_labels
    }

    _validate_required_labels(required_labels, head)

    for i in range(batch_size):
        img_path = image_paths[i]

        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Could not read image {img_path}")
        orig_h, orig_w = img.shape[:2]

        preds_for_image = _build_preds_for_image(
            head_output, required_labels, i
        )

        if _is_all_empty(preds_for_image, required_labels):
            yield {"file": str(img_path)}
            continue

        transformed = _prepare_transformed(
            preds_for_image,
            required_labels,
            head,
            orig_h,
            orig_w,
            train_size,
            keep_aspect_ratio,
        )
        yield from _emit_annotations(
            head, img_path, preds_for_image, transformed, required_labels
        )


def _validate_required_labels(
    required_labels: set[str], head: "lxt.nodes.BaseHead"
) -> None:
    for task in required_labels:
        if task not in ALLOWED_ANNOTATE_LABELS:
            raise ValueError(
                f"Unsupported task: {task}. Please create a custom annotate() method for head {head.name}."
            )


def _build_preds_for_image(
    head_output: Packet[Tensor], required_labels: set[str], i: int
) -> dict[str, Tensor]:
    return {
        task: (
            head_output["ocr"][i].unsqueeze(0)
            if task == "text"
            else head_output[task][i]
        )
        for task in required_labels
    }


def _is_all_empty(
    preds_for_image: dict[str, Tensor], required_labels: set[str]
) -> bool:
    non_text_labels = required_labels - {"text"}
    return bool(non_text_labels) and all(
        len(preds_for_image[task]) == 0 for task in non_text_labels
    )


def _prepare_transformed(
    preds_for_image: dict[str, Tensor],
    required_labels: set[str],
    head: "lxt.nodes.BaseHead",
    orig_h: int,
    orig_w: int,
    train_size: tuple[int, int],
    keep_aspect_ratio: bool,
) -> _Transformed:
    transformed: _Transformed = {}
    if "boundingbox" in required_labels:
        raw_boxes = (
            preds_for_image["boundingbox"][:, :4].detach().cpu().numpy()
        )
        transformed["norm_boxes"] = transform_boxes(
            raw_boxes, orig_h, orig_w, train_size, keep_aspect_ratio
        )
    if "keypoints" in required_labels:
        raw_kpts = preds_for_image["keypoints"].detach().cpu().float().numpy()
        transformed["norm_kpts"] = transform_keypoints(
            raw_kpts, orig_h, orig_w, train_size, keep_aspect_ratio
        )
    if "instance_segmentation" in required_labels:
        raw_masks = (
            preds_for_image["instance_segmentation"]
            .detach()
            .cpu()
            .float()
            .numpy()
        )
        transformed["norm_masks"] = transform_masks(
            raw_masks, orig_h, orig_w, train_size, keep_aspect_ratio
        )
    if "segmentation" in required_labels:
        bin_mask = (
            seg_output_to_bool(preds_for_image["segmentation"])
            .detach()
            .cpu()
            .float()
            .numpy()
        )
        transformed["norm_masks"] = transform_masks(
            bin_mask, orig_h, orig_w, train_size, keep_aspect_ratio
        )
    if "classification" in required_labels:
        transformed["pred_classes"] = (
            preds_for_image["classification"].detach().cpu().float().numpy()
        )
    if "text" in required_labels:
        if not hasattr(head, "decoder"):
            raise ValueError("Head does not have a decoder for text output.")
        transformed["pred_text"] = head.decoder(preds_for_image["text"])  # type: ignore
    return transformed


def _annotate_boundingbox(
    head: "lxt.nodes.BaseHead",
    img_path: Path,
    preds_for_image: dict[str, Tensor],
    transformed: _Transformed,
) -> DatasetIterator:
    assert "norm_boxes" in transformed
    norm_boxes = transformed["norm_boxes"]
    for idx, inst in enumerate(preds_for_image["boundingbox"]):
        x, y, w, h = norm_boxes[idx]
        yield {
            "file": str(img_path),
            "task_name": head.task_name,
            "annotation": {
                "instance_id": idx,
                "class": head.classes.inverse[int(inst[5].item())],
                "boundingbox": {"x": x, "y": y, "w": w, "h": h},
            },
        }


def _annotate_keypoints(
    head: "lxt.nodes.BaseHead",
    img_path: Path,
    transformed: _Transformed,
) -> DatasetIterator:
    assert "norm_kpts" in transformed
    for idx, pts in enumerate(transformed["norm_kpts"]):
        kps = [(float(x), float(y), round(v)) for x, y, v in pts]
        yield {
            "file": str(img_path),
            "task_name": head.task_name,
            "annotation": {
                "instance_id": idx,
                "keypoints": {"keypoints": kps},
            },
        }


def _annotate_instance_segmentation(
    head: "lxt.nodes.BaseHead",
    img_path: Path,
    transformed: _Transformed,
) -> DatasetIterator:
    assert "norm_masks" in transformed
    for idx, mask in enumerate(transformed["norm_masks"]):
        yield {
            "file": str(img_path),
            "task_name": head.task_name,
            "annotation": {
                "instance_id": idx,
                "instance_segmentation": {"mask": mask.astype(np.bool_)},
            },
        }


def _annotate_segmentation(
    head: "lxt.nodes.BaseHead",
    img_path: Path,
    transformed: _Transformed,
) -> DatasetIterator:
    assert "norm_masks" in transformed
    for idx, mask in enumerate(transformed["norm_masks"]):
        yield {
            "file": str(img_path),
            "task_name": head.task_name,
            "annotation": {
                "class": head.classes.inverse[idx],
                "segmentation": {"mask": mask.astype(np.bool_)},
            },
        }


def _annotate_classification(
    head: "lxt.nodes.BaseHead",
    img_path: Path,
    transformed: _Transformed,
) -> DatasetIterator:
    assert "pred_classes" in transformed
    yield {
        "file": str(img_path),
        "task_name": head.task_name,
        "annotation": {
            "class": head.classes.inverse[
                int(transformed["pred_classes"].argmax())
            ],
        },
    }


def _annotate_text(
    head: "lxt.nodes.BaseHead",
    img_path: Path,
    transformed: _Transformed,
) -> DatasetIterator:
    assert "pred_text" in transformed
    yield {
        "file": str(img_path),
        "task_name": head.task_name,
        "annotation": {"metadata": {"text": transformed["pred_text"][0][0]}},
    }


def _emit_annotations(
    head: "lxt.nodes.BaseHead",
    img_path: Path,
    preds_for_image: dict[str, Tensor],
    transformed: _Transformed,
    required_labels: set[str],
) -> DatasetIterator:
    for task in required_labels:
        if task == "boundingbox":
            yield from _annotate_boundingbox(
                head, img_path, preds_for_image, transformed
            )
        elif task == "keypoints":
            yield from _annotate_keypoints(head, img_path, transformed)
        elif task == "instance_segmentation":
            yield from _annotate_instance_segmentation(
                head, img_path, transformed
            )
        elif task == "segmentation":
            yield from _annotate_segmentation(head, img_path, transformed)
        elif task == "classification":
            yield from _annotate_classification(head, img_path, transformed)
        elif task == "text":
            yield from _annotate_text(head, img_path, transformed)
