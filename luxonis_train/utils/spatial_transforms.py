"""Maps boxes, keypoints, and masks from the resized image that the
model sees back to the original image.
"""

import cv2
import numpy as np


def compute_ratio_and_padding(
    orig_h: int,
    orig_w: int,
    train_size: tuple[int, int],
    keep_aspect_ratio: bool,
) -> tuple[float | None, float, float]:
    r"""Compute the scale and the padding of a letterbox resize.

    With ``keep_aspect_ratio``, the ratio is
    :math:`r = \min(h_t / h_o, w_t / w_o)`, where :math:`(h_t, w_t)` is
    ``train_size`` and :math:`(h_o, w_o)` is the original size. The
    padding is :math:`p_x = (w_t - r w_o) / 2` on the left and the
    right, and :math:`p_y = (h_t - r h_o) / 2` on the top and the
    bottom. Without ``keep_aspect_ratio``, the loader stretches the
    image, so the function returns no ratio and no padding.

    Args:
        orig_h (int): The height of the original image, in pixels.
        orig_w (int): The width of the original image, in pixels.
        train_size (tuple[int, int]): The height and the width of the
            model input.
        keep_aspect_ratio (bool): Whether the loader letterboxed the
            image.

    Returns:
        tuple[float | None, float, float]: The ratio :math:`r`, the
        padding :math:`p_x`, and the padding :math:`p_y`. Without
        ``keep_aspect_ratio``, the result is ``(None, 0, 0)``.

    Example:
        >>> from luxonis_train.utils.spatial_transforms import (
        ...     compute_ratio_and_padding,
        ... )
        >>> compute_ratio_and_padding(100, 200, (200, 200), True)
        (1.0, 0.0, 50.0)
        >>> compute_ratio_and_padding(100, 200, (200, 200), False)
        (None, 0, 0)

    """
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
    """Convert boxes from the model input to the original image.

    With ``keep_aspect_ratio``, the function subtracts the padding and
    divides by the ratio from `compute_ratio_and_padding`. Then it
    divides the coordinates by the original width and height. It does
    not clip the result, so a box in the padding gets values outside
    ``[0, 1]``.

    **Warning:** Without ``keep_aspect_ratio``, the function does not
    scale the boxes from ``train_size``. The result is correct only when
    the original size equals ``train_size``.

    Args:
        raw_boxes (``np.ndarray``): The boxes in the pixels of the model
            input, of shape ``[N, 4]``, in ``xyxy`` format.
        orig_h (int): The height of the original image, in pixels.
        orig_w (int): The width of the original image, in pixels.
        train_size (tuple[int, int]): The height and the width of the
            model input.
        keep_aspect_ratio (bool): Whether the loader letterboxed the
            image.

    Returns:
        ``np.ndarray``: The boxes of shape ``[N, 4]`` in normalized
        ``xywh`` format, where ``x`` and ``y`` give the top-left corner.
        An empty ``raw_boxes`` gives an empty array of shape ``[0]``.

    Example:
        >>> import numpy as np
        >>> from luxonis_train.utils import transform_boxes
        >>> boxes = np.array([[20.0, 60.0, 120.0, 110.0]])
        >>> transform_boxes(boxes, 100, 200, (200, 200), True).tolist()
        [[0.1, 0.1, 0.5, 0.5]]

    """
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
    """Convert keypoints from the model input to the original image.

    With ``keep_aspect_ratio``, the function subtracts the padding and
    divides by the ratio from `compute_ratio_and_padding`. Then it
    divides ``x`` by the original width and ``y`` by the original
    height. The third value does not change. The function does not clip
    the result.

    **Warning:** Without ``keep_aspect_ratio``, the function does not
    scale the keypoints from ``train_size``. The result is correct only
    when the original size equals ``train_size``.

    Args:
        raw_kpts (``np.ndarray``): The keypoints in the pixels of the
            model input, of shape ``[N, K, 3]``. The last axis holds
            ``x``, ``y``, and a visibility or score value.
        orig_h (int): The height of the original image, in pixels.
        orig_w (int): The width of the original image, in pixels.
        train_size (tuple[int, int]): The height and the width of the
            model input.
        keep_aspect_ratio (bool): Whether the loader letterboxed the
            image.

    Returns:
        ``np.ndarray``: The keypoints of shape ``[N, K, 3]``, as
        ``float64``, with normalized ``x`` and ``y``.

    Example:
        >>> import numpy as np
        >>> from luxonis_train.utils import transform_keypoints
        >>> kpts = np.array([[[100.0, 100.0, 2.0]]])
        >>> transform_keypoints(kpts, 100, 200, (200, 200), True).tolist()
        [[[0.5, 0.5, 2.0]]]

    """
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
    """Convert masks from the model input to the original image.

    With ``keep_aspect_ratio``, the function crops the padding from
    each mask. It truncates the crop bounds to integers. Then it resizes
    each mask to the original size with nearest-neighbour interpolation.

    Args:
        raw_masks (``np.ndarray``): The masks at the size of the model
            input, of shape ``[N, H, W]``. ``N`` must be at least ``1``,
            because ``np.stack`` rejects an empty list. The data type
            must be one that ``cv2.resize`` accepts, such as
            ``float32``.
        orig_h (int): The height of the original image, in pixels.
        orig_w (int): The width of the original image, in pixels.
        train_size (tuple[int, int]): The height and the width of the
            model input.
        keep_aspect_ratio (bool): Whether the loader letterboxed the
            image.

    Returns:
        ``np.ndarray``: The masks of shape ``[N, orig_h, orig_w]``.

    Example:
        >>> import numpy as np
        >>> from luxonis_train.utils import transform_masks
        >>> masks = np.zeros((1, 200, 200), dtype=np.float32)
        >>> masks[:, 50:150] = 1.0
        >>> result = transform_masks(masks, 100, 200, (200, 200), True)
        >>> result.shape, float(result.min())
        ((1, 100, 200), 1.0)

    """
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
