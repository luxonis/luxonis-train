from pathlib import Path

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

    for task in required_labels:
        if task not in ALLOWED_ANNOTATE_LABELS:
            raise ValueError(
                f"Unsupported task: {task}. Please create a custom annotate() method for head {head.name}."
            )

    for i in range(batch_size):
        img_path = image_paths[i]

        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Could not read image {img_path}")
        orig_h, orig_w = img.shape[:2]

        preds_for_image = {
            task: head_output["ocr"][i]
            if task == "text"
            else head_output[task][i]
            for task in required_labels
        }

        if all(
            len(preds_for_image[task]) == 0
            for task in required_labels
            if task != "text"
        ):
            yield {"file": str(img_path)}
            return

        if "boundingbox" in required_labels:
            raw_boxes = (
                preds_for_image["boundingbox"][:, :4].detach().cpu().numpy()
            )
            norm_boxes = transform_boxes(
                raw_boxes, orig_h, orig_w, train_size, keep_aspect_ratio
            )
        if "keypoints" in required_labels:
            raw_kpts = (
                preds_for_image["keypoints"].detach().cpu().float().numpy()
            )
            norm_kpts = transform_keypoints(
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
            norm_masks = transform_masks(
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
            norm_masks = transform_masks(
                bin_mask, orig_h, orig_w, train_size, keep_aspect_ratio
            )
        if "classification" in required_labels:
            pred_classes = (
                preds_for_image["classification"]
                .detach()
                .cpu()
                .float()
                .numpy()
            )
        if "text" in required_labels:
            if not hasattr(head, "decoder"):
                raise ValueError(
                    "Head does not have a decoder for text output."
                )
            pred_text = head.decoder(preds_for_image["text"])  # type: ignore

        for task in required_labels:
            if task == "boundingbox":
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
            elif task == "keypoints":
                for idx, pts in enumerate(norm_kpts):
                    kps = [(float(x), float(y), round(v)) for x, y, v in pts]
                    yield {
                        "file": str(img_path),
                        "task_name": head.task_name,
                        "annotation": {
                            "instance_id": idx,
                            "keypoints": {"keypoints": kps},
                        },
                    }
            elif task == "instance_segmentation":
                for idx, mask in enumerate(norm_masks):
                    yield {
                        "file": str(img_path),
                        "task_name": head.task_name,
                        "annotation": {
                            "instance_id": idx,
                            "instance_segmentation": {
                                "mask": mask.astype(np.bool_)
                            },
                        },
                    }
            elif task == "segmentation":
                for idx, mask in enumerate(norm_masks):
                    yield {
                        "file": str(img_path),
                        "task_name": head.task_name,
                        "annotation": {
                            "class": head.classes.inverse[idx],
                            "segmentation": {"mask": mask.astype(np.bool_)},
                        },
                    }
            elif task == "classification":
                yield {
                    "file": str(img_path),
                    "task_name": head.task_name,
                    "annotation": {
                        "class": head.classes.inverse[
                            int(pred_classes[i].argmax())
                        ],
                    },
                }
            elif task == "text":
                yield {
                    "file": str(img_path),
                    "task_name": head.task_name,
                    "annotation": {
                        "metadata": {
                            "text": pred_text[i][0],
                        }
                    },
                }
