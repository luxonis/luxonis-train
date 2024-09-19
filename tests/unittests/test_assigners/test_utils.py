import torch

from luxonis_train.assigners.utils import (
    batch_iou,
    candidates_in_gt,
    fix_collisions,
)


def test_fix_collisions():
    batch_size = 2
    n_max_boxes = 3
    n_anchors = 4

    mask_pos = torch.randint(0, 2, (batch_size, n_max_boxes, n_anchors))
    overlaps = torch.rand(batch_size, n_max_boxes, n_anchors)

    assigned_gt_idx, mask_pos_sum, new_mask_pos = fix_collisions(
        mask_pos, overlaps, n_max_boxes
    )

    assert assigned_gt_idx.shape == (batch_size, n_anchors)
    assert mask_pos_sum.shape == (batch_size, n_anchors)
    assert new_mask_pos.shape == (batch_size, n_max_boxes, n_anchors)


def test_candidates_in_gt():
    n_anchors = 4
    batch_size = 2
    n_max_boxes = 3

    anchor_centers = torch.rand(n_anchors, 2)
    gt_bboxes = torch.rand(batch_size * n_max_boxes, 4)

    candidates = candidates_in_gt(anchor_centers, gt_bboxes)

    assert candidates.shape == (batch_size * n_max_boxes, n_anchors)
    assert candidates.dtype == torch.float32


def test_batch_iou():
    batch_size = 2
    n = 3
    m = 4

    batch1 = torch.rand(batch_size, n, 4)
    batch2 = torch.rand(batch_size, m, 4)

    ious = batch_iou(batch1, batch2)

    assert ious.shape == (batch_size, n, m)
    assert ious.dtype == torch.float32
