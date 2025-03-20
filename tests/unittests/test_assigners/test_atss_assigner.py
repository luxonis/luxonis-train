import torch

from luxonis_train.assigners import ATSSAssigner


def test_init():
    assigner = ATSSAssigner(n_classes=80, topk=9)
    assert assigner.n_classes == 80
    assert assigner.topk == 9


def test_forward():
    bs = 10
    n_max_boxes = 5
    n_anchors = 100
    n_classes = 80
    topk = 9

    assigner = ATSSAssigner(n_classes=n_classes, topk=topk)
    anchor_bboxes = torch.rand(n_anchors, 4)
    n_level_bboxes = [20, 30, 50]
    gt_labels = torch.rand(bs, n_max_boxes, 1)
    gt_bboxes = torch.zeros(bs, n_max_boxes, 4)
    mask_gt = torch.rand(bs, n_max_boxes, 1)
    pred_bboxes = torch.rand(bs, n_anchors, 4)

    labels, bboxes, scores, mask, assigned_gt_idx = assigner.forward(
        anchor_bboxes,
        n_level_bboxes,
        gt_labels,
        gt_bboxes,
        mask_gt,
        pred_bboxes,
    )

    assert labels.shape == (bs, n_anchors)
    assert bboxes.shape == (bs, n_anchors, 4)
    assert scores.shape == (bs, n_anchors, n_classes)
    assert mask.shape == (bs, n_anchors)
    assert assigned_gt_idx.shape == (bs, n_anchors)


def test_get_bbox_center():
    assigner = ATSSAssigner(n_classes=80, topk=9)
    bbox = torch.tensor([[0, 0, 10, 10], [10, 10, 20, 20]])
    centers = assigner._get_bbox_center(bbox)
    expected_centers = torch.tensor([[5, 5], [15, 15]])
    assert torch.all(torch.eq(centers, expected_centers))


def test_select_topk_candidates():
    batch_size = 2
    n_max_boxes = 3
    n_anchors = 10
    topk = 2
    n_level_bboxes = [4, 6]  # Mock number of boxes per level

    assigner = ATSSAssigner(n_classes=80, topk=topk)
    distances = torch.rand(batch_size, n_max_boxes, n_anchors)
    mask_gt = torch.ones(batch_size, n_max_boxes, 1)

    is_in_topk, topk_idxs = assigner._select_topk_candidates(
        distances, n_level_bboxes, mask_gt
    )

    assert is_in_topk.shape == (batch_size, n_max_boxes, n_anchors)
    assert topk_idxs.shape == (
        batch_size,
        n_max_boxes,
        topk * len(n_level_bboxes),
    )


def test_get_positive_samples():
    batch_size = 2
    n_max_boxes = 3
    n_anchors = 10
    topk = 2

    assigner = ATSSAssigner(n_classes=80, topk=topk)
    assigner.bs = batch_size
    assigner.n_max_boxes = n_max_boxes
    assigner.n_anchors = n_anchors
    is_in_topk = torch.rand(batch_size, n_max_boxes, n_anchors)
    topk_idxs = torch.randint(0, n_anchors, (batch_size, n_max_boxes, topk))
    overlaps = torch.rand(batch_size, n_max_boxes, n_anchors)

    is_pos = assigner._get_positive_samples(is_in_topk, topk_idxs, overlaps)

    assert is_pos.shape == (batch_size, n_max_boxes, n_anchors)


def test_get_final_assignments():
    batch_size = 2
    n_max_boxes = 3
    n_anchors = 10
    n_classes = 80

    assigner = ATSSAssigner(n_classes=n_classes, topk=9)
    assigner.bs = batch_size
    assigner.n_anchors = n_anchors
    assigner.n_max_boxes = n_max_boxes

    gt_labels = torch.randint(0, n_classes, (batch_size, n_max_boxes, 1))
    gt_bboxes = torch.rand(batch_size, n_max_boxes, 4)
    assigned_gt_idx = torch.randint(0, n_max_boxes, (batch_size, n_anchors))
    mask_pos_sum = torch.randint(0, 2, (batch_size, n_anchors))

    (assigned_labels, assigned_bboxes, assigned_scores) = (
        assigner._get_final_assignments(
            gt_labels, gt_bboxes, assigned_gt_idx, mask_pos_sum
        )
    )

    assert assigned_labels.shape == (batch_size, n_anchors)
    assert assigned_bboxes.shape == (batch_size, n_anchors, 4)
    assert assigned_scores.shape == (batch_size, n_anchors, n_classes)
