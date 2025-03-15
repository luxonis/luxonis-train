import torch

from luxonis_train.assigners import TaskAlignedAssigner


def test_init():
    assigner = TaskAlignedAssigner(
        n_classes=80, topk=13, alpha=1.0, beta=6.0, eps=1e-9
    )
    assert assigner.n_classes == 80
    assert assigner.topk == 13
    assert assigner.alpha == 1.0
    assert assigner.beta == 6.0
    assert assigner.eps == 1e-9


def test_forward():
    batch_size = 10
    n_anchors = 100
    n_max_boxes = 5
    n_classes = 80

    assigner = TaskAlignedAssigner(n_classes=n_classes, topk=13)

    # Create mock inputs
    pred_scores = torch.rand(batch_size, n_anchors, 1)
    pred_bboxes = torch.rand(batch_size, n_anchors, 4)
    anchor_points = torch.rand(n_anchors, 2)
    gt_labels = torch.rand(batch_size, n_max_boxes, 1)
    gt_bboxes = torch.zeros(batch_size, n_max_boxes, 4)  # no gt bboxes
    mask_gt = torch.rand(batch_size, n_max_boxes, 1)

    labels, bboxes, scores, mask, assigned_gt_idx = assigner.forward(
        pred_scores, pred_bboxes, anchor_points, gt_labels, gt_bboxes, mask_gt
    )

    assert labels.shape == (batch_size, n_anchors)
    assert bboxes.shape == (batch_size, n_anchors, 4)
    assert scores.shape == (batch_size, n_anchors, n_classes)
    assert mask.shape == (batch_size, n_anchors)
    assert assigned_gt_idx.shape == (batch_size, n_anchors)

    # Labels should be `n_classes` as there are no GT boxes
    assert labels.unique().tolist() == [n_classes]

    # All results should be zero as there are no GT boxes
    assert torch.equal(bboxes, torch.zeros_like(bboxes))
    assert torch.equal(scores, torch.zeros_like(scores))
    assert torch.equal(mask, torch.zeros_like(mask))
    assert torch.equal(assigned_gt_idx, torch.zeros_like(assigned_gt_idx))


def test_get_alignment_metric():
    batch_size = 2
    n_anchors = 5
    n_max_boxes = 3
    n_classes = 80

    pred_scores = torch.rand(batch_size, n_anchors, n_classes)
    pred_bboxes = torch.rand(batch_size, n_anchors, 4)
    gt_labels = torch.randint(0, n_classes, (batch_size, n_max_boxes, 1))
    gt_bboxes = torch.rand(batch_size, n_max_boxes, 4)

    assigner = TaskAlignedAssigner(
        n_classes=n_classes, topk=13, alpha=1.0, beta=6.0, eps=1e-9
    )
    assigner.bs = pred_scores.size(0)
    assigner.n_max_boxes = gt_bboxes.size(1)

    align_metric, overlaps = assigner._get_alignment_metric(
        pred_scores, pred_bboxes, gt_labels, gt_bboxes
    )

    assert align_metric.shape == (batch_size, n_max_boxes, n_anchors)
    assert overlaps.shape == (batch_size, n_max_boxes, n_anchors)
    assert align_metric.dtype == torch.float32
    assert overlaps.dtype == torch.float32
    assert align_metric.min() >= 0
    assert align_metric.max() <= 1
    assert overlaps.min() >= 0
    assert overlaps.max() <= 1


def test_select_topk_candidates():
    batch_size = 2
    n_max_boxes = 3
    n_anchors = 5
    topk = 2

    metrics = torch.rand(batch_size, n_max_boxes, n_anchors)
    mask_gt = torch.rand(batch_size, n_max_boxes, 1)

    assigner = TaskAlignedAssigner(n_classes=80, topk=topk)

    is_in_topk = assigner._select_topk_candidates(metrics)
    topk_mask = mask_gt.repeat([1, 1, topk]).bool()
    assert torch.equal(
        assigner._select_topk_candidates(metrics),
        assigner._select_topk_candidates(metrics, topk_mask=topk_mask),
    )
    assert is_in_topk.shape == (batch_size, n_max_boxes, n_anchors)
    assert is_in_topk.dtype == torch.float32

    assert is_in_topk.sum(dim=-1).max() <= topk


def test_get_final_assignments():
    batch_size = 2
    n_max_boxes = 3
    n_anchors = 5
    n_classes = 80

    gt_labels = torch.randint(0, n_classes, (batch_size, n_max_boxes, 1))
    gt_bboxes = torch.rand(batch_size, n_max_boxes, 4)
    assigned_gt_idx = torch.randint(0, n_max_boxes, (batch_size, n_anchors))
    mask_pos_sum = torch.randint(0, 2, (batch_size, n_anchors))

    assigner = TaskAlignedAssigner(n_classes=n_classes, topk=13)
    assigner.bs = batch_size  # Set batch size
    assigner.n_max_boxes = gt_bboxes.size(1)

    (assigned_labels, assigned_bboxes, assigned_scores) = (
        assigner._get_final_assignments(
            gt_labels, gt_bboxes, assigned_gt_idx, mask_pos_sum
        )
    )

    assert assigned_labels.shape == (batch_size, n_anchors)
    assert assigned_bboxes.shape == (batch_size, n_anchors, 4)
    assert assigned_scores.shape == (batch_size, n_anchors, n_classes)
    assert assigned_labels.min() >= 0
    assert assigned_labels.max() <= n_classes
