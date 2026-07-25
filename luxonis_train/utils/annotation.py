from pathlib import Path
from typing import Any

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


def default_annotate(
    head: "lxt.nodes.BaseHead",
    head_output: Packet[Tensor],
    image_paths: list[Path],
    config_preprocessing: PreprocessingConfig,
) -> DatasetIterator:
    """Convert head output to a DatasetIterator for annotations in a
    format suitable for LuxonisDataset.

    @type head: BaseHead
    @param head: The head from which to extract annotations.
    @type head_output: Packet[Tensor]
    @param head_output: The output from the head containing predictions.
    @type image_paths: list[Path]
    @param image_paths: List of paths to the images corresponding to the
        head output.
    @type config_preprocessing: PreprocessingConfig
    @param config_preprocessing: Preprocessing configuration containing
        image size and aspect ratio settings.
    @rtype: DatasetIterator
    @return: A DatasetIterator yielding annotations for each image.
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
            return

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
            head, img_path, preds_for_image, transformed, i, required_labels
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
        task: head_output["ocr"][i] if task == "text" else head_output[task][i]
        for task in required_labels
    }


def _is_all_empty(
    preds_for_image: dict[str, Tensor], required_labels: set[str]
) -> bool:
    return all(
        len(preds_for_image[task]) == 0
        for task in required_labels
        if task != "text"
    )


def _prepare_transformed(
    preds_for_image: dict[str, Tensor],
    required_labels: set[str],
    head: "lxt.nodes.BaseHead",
    orig_h: int,
    orig_w: int,
    train_size: tuple[int, int],
    keep_aspect_ratio: bool,
) -> dict[str, Any]:
    """Convert raw per-task predictions into normalized annotations."""
    transformed: dict[str, Any] = {}
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
    transformed: dict[str, Any],
    i: int,
) -> DatasetIterator:
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
    preds_for_image: dict[str, Tensor],
    transformed: dict[str, Any],
    i: int,
) -> DatasetIterator:
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
    preds_for_image: dict[str, Tensor],
    transformed: dict[str, Any],
    i: int,
) -> DatasetIterator:
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
    preds_for_image: dict[str, Tensor],
    transformed: dict[str, Any],
    i: int,
) -> DatasetIterator:
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
    preds_for_image: dict[str, Tensor],
    transformed: dict[str, Any],
    i: int,
) -> DatasetIterator:
    yield {
        "file": str(img_path),
        "task_name": head.task_name,
        "annotation": {
            "class": head.classes.inverse[
                int(transformed["pred_classes"][i].argmax())
            ],
        },
    }


def _annotate_text(
    head: "lxt.nodes.BaseHead",
    img_path: Path,
    preds_for_image: dict[str, Tensor],
    transformed: dict[str, Any],
    i: int,
) -> DatasetIterator:
    yield {
        "file": str(img_path),
        "task_name": head.task_name,
        "annotation": {"metadata": {"text": transformed["pred_text"][i][0]}},
    }


_ANNOTATORS = {
    "boundingbox": _annotate_boundingbox,
    "keypoints": _annotate_keypoints,
    "instance_segmentation": _annotate_instance_segmentation,
    "segmentation": _annotate_segmentation,
    "classification": _annotate_classification,
    "text": _annotate_text,
}


def _emit_annotations(
    head: "lxt.nodes.BaseHead",
    img_path: Path,
    preds_for_image: dict[str, Tensor],
    transformed: dict[str, Any],
    i: int,
    required_labels: set[str],
) -> DatasetIterator:
    for task in required_labels:
        annotator = _ANNOTATORS.get(task)
        if annotator is not None:
            yield from annotator(
                head, img_path, preds_for_image, transformed, i
            )
