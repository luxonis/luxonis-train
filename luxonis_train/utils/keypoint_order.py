"""Image-relative ordering of four-corner keypoint annotations."""

import numpy as np


def order_quad_keypoints(
    keypoints: np.ndarray, image_height: int, image_width: int
) -> np.ndarray:
    """Order normalized ``[N, 12]`` corner triplets clockwise in the
    image.

    Start at minimum pixel-space x+y, breaking ties by y, then x.
    Positive atan2 order is clockwise because image y increases
    downward. Return a new array with the same shape/dtype and
    visibility attached to each corner. Instance order is unchanged.
    Coordinates must be finite, known corners of a convex quadrilateral;
    visibility may be zero. Clipped or coincident corners are sorted
    deterministically using their available coordinates, but lost
    geometry cannot be recovered.

    """
    if keypoints.ndim != 2 or keypoints.shape[1] != 12:
        raise ValueError(
            "image_clockwise requires exactly four keypoints per instance "
            f"(shape [N, 12]); got {keypoints.shape}."
        )
    if image_height <= 0 or image_width <= 0:
        raise ValueError("Image dimensions must be positive.")
    corners = keypoints.reshape(-1, 4, 3)
    xy = corners[..., :2].astype(np.float64) * [image_width, image_height]
    if not np.isfinite(xy).all():
        raise ValueError("image_clockwise requires finite corner coordinates.")
    if not len(corners):
        return keypoints.copy()

    offsets = xy - xy.mean(axis=1, keepdims=True)
    angles = np.arctan2(offsets[..., 1], offsets[..., 0])
    # Coordinate tie-breaks make degenerate/clipped corners deterministic.
    clockwise = np.lexsort((xy[..., 0], xy[..., 1], angles), axis=1)
    start = np.lexsort((xy[..., 0], xy[..., 1], xy.sum(axis=-1)), axis=1)[:, 0]
    start_position = (clockwise == start[:, None]).argmax(axis=1)
    indices = (start_position[:, None] + np.arange(4)) % 4
    permutation = np.take_along_axis(clockwise, indices, axis=1)
    return np.take_along_axis(corners, permutation[..., None], axis=1).reshape(
        keypoints.shape
    )
