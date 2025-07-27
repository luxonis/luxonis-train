import cv2
import numpy as np


def compute_ratio_and_padding(
    orig_h: int,
    orig_w: int,
    train_size: tuple[int, int],
    keep_aspect_ratio: bool,
) -> tuple[float | None, float, float]:
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
    train_size: tuple[int, int],
    keep_aspect_ratio: bool,
) -> np.ndarray:
    """Transforms raw bounding boxes to normalized coordinates based on
    the original image size and training size."""
    ratio, pad_x, pad_y = compute_ratio_and_padding(
        orig_h, orig_w, train_size, keep_aspect_ratio
    )
    boxes = []
    for x1, y1, x2, y2 in raw_boxes:
        if ratio is not None:
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
    train_size: tuple[int, int],
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
            if ratio is not None:
                x = (x - pad_x) / ratio
                y = (y - pad_y) / ratio
            out[i, j] = (x / orig_w, y / orig_h, float(v))
    return out


def transform_masks(
    raw_masks: np.ndarray,
    orig_h: int,
    orig_w: int,
    train_size: tuple[int, int],
    keep_aspect_ratio: bool,
) -> np.ndarray:
    """Transforms raw masks to normalized size based on the original
    image size and training size."""
    ratio, pad_x, pad_y = compute_ratio_and_padding(
        orig_h, orig_w, train_size, keep_aspect_ratio
    )
    norm_masks = []
    for mask in raw_masks:
        if ratio is not None:
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
