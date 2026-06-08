import numpy as np

from luxonis_train.utils.spatial_transforms import (
    compute_ratio_and_padding,
    transform_boxes,
    transform_keypoints,
    transform_masks,
)


def test_compute_ratio_and_padding_with_and_without_aspect_ratio():
    assert compute_ratio_and_padding(10, 20, (20, 20), True) == (1.0, 0.0, 5.0)
    assert compute_ratio_and_padding(10, 20, (20, 20), False) == (
        None,
        0,
        0,
    )


def test_transform_boxes_with_and_without_aspect_ratio():
    raw_boxes = np.array([[5.0, 6.0, 15.0, 11.0]])

    keep_aspect = transform_boxes(raw_boxes, 10, 20, (20, 20), True)
    stretched = transform_boxes(raw_boxes, 10, 20, (20, 20), False)

    np.testing.assert_allclose(keep_aspect, [[0.25, 0.1, 0.5, 0.5]])
    np.testing.assert_allclose(stretched, [[0.25, 0.6, 0.5, 0.5]])


def test_transform_keypoints_with_and_without_aspect_ratio():
    raw_keypoints = np.array([[[5.0, 6.0, 2.0], [15.0, 11.0, 1.0]]])

    keep_aspect = transform_keypoints(raw_keypoints, 10, 20, (20, 20), True)
    stretched = transform_keypoints(raw_keypoints, 10, 20, (20, 20), False)

    np.testing.assert_allclose(
        keep_aspect, [[[0.25, 0.1, 2.0], [0.75, 0.6, 1.0]]]
    )
    np.testing.assert_allclose(
        stretched, [[[0.25, 0.6, 2.0], [0.75, 1.1, 1.0]]]
    )


def test_transform_masks_with_and_without_aspect_ratio():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:15, 5:15] = 1

    keep_aspect = transform_masks(np.array([mask]), 10, 20, (20, 20), True)
    stretched = transform_masks(np.array([mask]), 10, 20, (20, 20), False)

    assert keep_aspect.shape == (1, 10, 20)
    assert stretched.shape == (1, 10, 20)
    assert keep_aspect.dtype == np.uint8
    assert stretched.dtype == np.uint8
    assert keep_aspect.sum() > 0
    assert stretched.sum() > 0
