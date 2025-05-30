from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal, Tuple

import cv2
import numpy as np
import torch
from luxonis_ml.data import DatasetIterator, LuxonisDataset
from luxonis_ml.typing import PathType

import luxonis_train as lxt
from luxonis_train import tasks
from luxonis_train.attached_modules.visualizers.utils import seg_output_to_bool

from .infer_utils import create_loader_from_directory


def annotate_from_directory(
    model: "lxt.LuxonisModel",
    img_paths: Iterable[PathType],
    dataset_name: str,
    bucket_storage: Literal["local", "gcs"] = "local",
    delete_local: bool = True,
    delete_remote: bool = True,
    team_id: str | None = None,
    head_name: str | None = None,
) -> LuxonisDataset:
    """Annotate images from a directory using the specified model and
    create a LuxonisDataset.

    @param model: The LuxonisModel to use for annotation.
    @type model: lxt.LuxonisModel
    @param img_paths: Iterable of image paths to annotate.
    @type img_paths: Iterable[PathType]
    @param dataset_name: Name of the dataset to create.
    @type dataset_name: str
    @param bucket_storage: Storage type for the dataset, either 'local'
        or 'gcs'.
    @type bucket_storage: Literal['local', 'gcs']
    @param delete_local: Whether to delete local files after processing.
    @type delete_local: bool
    @param delete_remote: Whether to delete remote files after
        processing.
    @type delete_remote: bool
    @param team_id: Optional team ID for the dataset.
    @type team_id: str | None
    @param head_name: The name of the head to use for annotation;
        required when multiple heads are present.
    @type head_name: str | None
    """

    img_paths = list(img_paths)

    loader = create_loader_from_directory(img_paths, model)

    annotated_dataset = LuxonisDataset(
        dataset_name=dataset_name,
        bucket_storage=bucket_storage,
        delete_local=delete_local,
        delete_remote=delete_remote,
        team_id=team_id,
    )

    generator = annotated_dataset_generator(
        model, loader, img_paths, head_name
    )
    annotated_dataset.add(generator)
    annotated_dataset.make_splits()

    loader.dataset.dataset.delete_dataset()

    return annotated_dataset


def annotated_dataset_generator(
    model, loader, img_paths: Iterable[Path], head_name: str | None = None
) -> DatasetIterator:
    """Generator that yields annotations for images processed by the
    model."""
    img_paths = list(img_paths)
    train_size = model.cfg_preprocessing.train_image_size
    keep_ar = model.cfg_preprocessing.keep_aspect_ratio

    lm = model.lightning_module.eval()
    sample = next(iter(loader))[0]
    with torch.no_grad():
        sample_out = lm(sample)
    heads = list(sample_out.outputs.keys())
    if len(heads) > 1 and head_name is None:
        raise ValueError("Multiple heads detected; specify head_name")
    head = head_name or heads[0]
    node = lm.nodes[head]

    for batch in loader:
        imgs, metas = batch
        with torch.no_grad():
            batch_out = lm(imgs).outputs[head]
        for i, raw_meta in enumerate(metas["/metadata/path"]):
            img_path = Path("".join(chr(int(c.item())) for c in raw_meta))
            preds_for_image = {
                task: batch_out[task][i] for task in node.task.required_labels
            }
            yield from process_single_image(
                preds_for_image, img_path, node, train_size, keep_ar
            )


def compute_ratio_and_padding(
    orig_h: int,
    orig_w: int,
    train_size: Tuple[int, int],
    keep_aspect_ratio: bool,
) -> Tuple[float, float, float]:
    """Computes the ratio and padding needed to transform bounding
    boxes, keypoints, and masks."""
    train_h, train_w = train_size
    if keep_aspect_ratio:
        ratio = min(train_h / orig_h, train_w / orig_w)
        pad_y = (train_h - orig_h * ratio) / 2
        pad_x = (train_w - orig_w * ratio) / 2
    else:
        ratio = None
        pad_y = pad_x = 0
    return ratio, pad_x, pad_y


def transform_boxes(
    raw_boxes: np.ndarray,
    orig_h: int,
    orig_w: int,
    train_size: Tuple[int, int],
    keep_aspect_ratio: bool,
) -> np.ndarray:
    """Transforms raw bounding boxes to normalized coordinates based on
    the original image size and training size."""
    ratio, pad_x, pad_y = compute_ratio_and_padding(
        orig_h, orig_w, train_size, keep_aspect_ratio
    )
    boxes = []
    for x1, y1, x2, y2 in raw_boxes:
        if keep_aspect_ratio:
            ox1 = (x1 - pad_x) / ratio
            oy1 = (y1 - pad_y) / ratio
            ow = (x2 - x1) / ratio
            oh = (y2 - y1) / ratio
        else:
            ox1, oy1 = x1, y1
            ow, oh = x2 - x1, y2 - y1
        boxes.append([ox1 / orig_w, oy1 / orig_h, ow / orig_w, oh / orig_h])
    return np.array(boxes, dtype=float)


def transform_keypoints(
    raw_kpts: np.ndarray,
    orig_h: int,
    orig_w: int,
    train_size: Tuple[int, int],
    keep_aspect_ratio: bool,
) -> np.ndarray:
    """Transforms raw keypoints to normalized coordinates based on the
    original image size and training size."""
    ratio, pad_x, pad_y = compute_ratio_and_padding(
        orig_h, orig_w, train_size, keep_aspect_ratio
    )
    N, K, _ = raw_kpts.shape
    out = np.zeros((N, K, 3), dtype=float)
    for i in range(N):
        for j in range(K):
            x, y, v = raw_kpts[i, j]
            if keep_aspect_ratio:
                x = (x - pad_x) / ratio
                y = (y - pad_y) / ratio
            out[i, j] = (x / orig_w, y / orig_h, float(v))
    return out


def transform_masks(
    raw_masks: np.ndarray,
    orig_h: int,
    orig_w: int,
    train_size: Tuple[int, int],
    keep_aspect_ratio: bool,
) -> np.ndarray:
    """Transforms raw masks to normalized size based on the original
    image size and training size."""
    ratio, pad_x, pad_y = compute_ratio_and_padding(
        orig_h, orig_w, train_size, keep_aspect_ratio
    )
    norm_masks = []
    for mask in raw_masks:
        if keep_aspect_ratio:
            y1 = int(pad_y)
            y2 = int(pad_y + orig_h * ratio)
            x1 = int(pad_x)
            x2 = int(pad_x + orig_w * ratio)
            m_cropped = mask[y1:y2, x1:x2]
        else:
            m_cropped = mask
        m_resized = cv2.resize(
            m_cropped, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
        )
        norm_masks.append(m_resized)
    return np.stack(norm_masks, axis=0)


def process_single_image(
    preds: dict[str, Any],
    img_path: Path,
    node: Any,
    train_size: Tuple[int, int],
    keep_aspect_ratio: bool,
) -> Iterable[dict[str, Any]]:
    """Processes a single image and yields annotations based on the
    model predictions."""
    if all(len(preds[task]) == 0 for task in node.task.required_labels):
        yield {"file": str(img_path)}
        return

    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image {img_path}")
    orig_h, orig_w = img.shape[:2]

    if "boundingbox" in node.task.required_labels:
        raw_boxes = preds["boundingbox"][:, :4].detach().cpu().numpy()
        norm_boxes = transform_boxes(
            raw_boxes, orig_h, orig_w, train_size, keep_aspect_ratio
        )
    if "keypoints" in node.task.required_labels:
        raw_kpts = preds["keypoints"].detach().cpu().float().numpy()
        norm_kpts = transform_keypoints(
            raw_kpts, orig_h, orig_w, train_size, keep_aspect_ratio
        )
    if "instance_segmentation" in node.task.required_labels:
        raw_masks = (
            preds["instance_segmentation"].detach().cpu().float().numpy()
        )
        norm_masks = transform_masks(
            raw_masks, orig_h, orig_w, train_size, keep_aspect_ratio
        )
    if "segmentation" in node.task.required_labels:
        bin_mask = (
            seg_output_to_bool(preds["segmentation"])
            .detach()
            .cpu()
            .float()
            .numpy()
        )
        norm_masks = transform_masks(
            bin_mask, orig_h, orig_w, train_size, keep_aspect_ratio
        )

    for task in node.task.required_labels:
        if task == "boundingbox":
            for idx, inst in enumerate(preds["boundingbox"]):
                x, y, w, h = norm_boxes[idx]
                yield {
                    "file": str(img_path),
                    "task_name": node.task_name,
                    "annotation": {
                        "instance_id": idx,
                        "class": node.classes.inverse[int(inst[5].item())],
                        "boundingbox": {"x": x, "y": y, "w": w, "h": h},
                    },
                }
        elif task == "keypoints":
            for idx, pts in enumerate(norm_kpts):
                kps = [(float(x), float(y), int(round(v))) for x, y, v in pts]
                yield {
                    "file": str(img_path),
                    "task_name": node.task_name,
                    "annotation": {
                        "instance_id": idx,
                        "keypoints": {"keypoints": kps},
                    },
                }
        elif task == "instance_segmentation":
            for idx, mask in enumerate(norm_masks):
                yield {
                    "file": str(img_path),
                    "task_name": node.task_name,
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
                    "task_name": node.task_name,
                    "annotation": {
                        "class": node.classes.inverse[idx],
                        "segmentation": {"mask": mask.astype(np.bool_)},
                    },
                }
