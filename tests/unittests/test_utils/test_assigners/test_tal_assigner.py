import torch

from luxonis_train.utils.assigners.tal_assigner import TaskAlignedAssigner


def test_init():
    assigner = TaskAlignedAssigner(n_classes=80, topk=13, alpha=1.0, beta=6.0, eps=1e-9)
    assert assigner.n_classes == 80
    assert assigner.topk == 13
    assert assigner.alpha == 1.0
    assert assigner.beta == 6.0
    assert assigner.eps == 1e-9


def test_forward():
    # Constants for clarity
    batch_size = 10
    num_anchors = 100
    num_max_boxes = 5
    num_classes = 80

    # Initialize the TaskAlignedAssigner
    assigner = TaskAlignedAssigner(n_classes=num_classes, topk=13)

    # Create mock inputs
    pred_scores = torch.rand(batch_size, num_anchors, 1)
    pred_bboxes = torch.rand(batch_size, num_anchors, 4)
    anchor_points = torch.rand(num_anchors, 2)
    gt_labels = torch.rand(batch_size, num_max_boxes, 1)
    gt_bboxes = torch.zeros(batch_size, num_max_boxes, 4)  # no gt bboxes
    mask_gt = torch.rand(batch_size, num_max_boxes, 1)

    # Call the forward method
    labels, bboxes, scores, mask = assigner.forward(
        pred_scores, pred_bboxes, anchor_points, gt_labels, gt_bboxes, mask_gt
    )

    # Assert the expected outcomes
    assert labels.shape == (batch_size, num_anchors)
    assert labels.unique().tolist() == [
        num_classes
    ]  # All labels should be num_classes as there are no GT boxes
    assert bboxes.shape == (batch_size, num_anchors, 4)
    assert torch.equal(
        bboxes, torch.zeros_like(bboxes)
    )  # All bboxes should be zero as there are no GT boxes
    assert (
        scores.shape
        == (
            batch_size,
            num_anchors,
            num_classes,
        )
    )  # TODO: We have this in doc string: Returns: ... assigned scores of shape [bs, n_anchors, 1],
    # it returns tensor of shape [bs, n_anchors, n_classes] instead
    assert torch.equal(
        scores, torch.zeros_like(scores)
    )  # All scores should be zero as there are no GT boxes
    assert mask.shape == (batch_size, num_anchors)
    assert torch.equal(
        mask, torch.zeros_like(mask)
    )  # All mask values should be zero as there are no GT boxes


def test_get_alignment_metric():
    # Create mock inputs
    bs = 2  # batch size
    n_anchors = 5
    n_max_boxes = 3
    n_classes = 80

    pred_scores = torch.rand(
        bs, n_anchors, n_classes
    )  # TODO: Same issue: works with n_classes instead of 1, change it in the doc string in the method itself!!!
    pred_bboxes = torch.rand(bs, n_anchors, 4)
    gt_labels = torch.randint(0, n_classes, (bs, n_max_boxes, 1))
    gt_bboxes = torch.rand(bs, n_max_boxes, 4)

    # Initialize the TaskAlignedAssigner
    assigner = TaskAlignedAssigner(
        n_classes=n_classes, topk=13, alpha=1.0, beta=6.0, eps=1e-9
    )
    assigner.bs = pred_scores.size(0)
    assigner.n_max_boxes = gt_bboxes.size(1)

    # Call the method
    align_metric, overlaps = assigner._get_alignment_metric(
        pred_scores, pred_bboxes, gt_labels, gt_bboxes
    )

    # Assert the expected outcomes
    assert align_metric.shape == (bs, n_max_boxes, n_anchors)
    assert overlaps.shape == (bs, n_max_boxes, n_anchors)
    assert align_metric.dtype == torch.float32
    assert overlaps.dtype == torch.float32
    assert (align_metric >= 0).all() and (
        align_metric <= 1
    ).all()  # Alignment metric should be in the range [0, 1]
    assert (overlaps >= 0).all() and (
        overlaps <= 1
    ).all()  # IoU should be in the range [0, 1]


def test_select_topk_candidates():
    # Constants for the test
    batch_size = 2
    num_max_boxes = 3
    num_anchors = 5
    topk = 2

    metrics = torch.rand(batch_size, num_max_boxes, num_anchors)
    mask_gt = torch.rand(batch_size, num_max_boxes, 1)

    # Initialize the TaskAlignedAssigner
    assigner = TaskAlignedAssigner(n_classes=80, topk=topk)

    # Call the method
    is_in_topk = assigner._select_topk_candidates(
        metrics,
    )
    topk_mask = mask_gt.repeat([1, 1, topk]).bool()
    assert torch.equal(
        assigner._select_topk_candidates(metrics),
        assigner._select_topk_candidates(metrics, topk_mask=topk_mask),
    )
    # Assert the expected outcomes
    assert is_in_topk.shape == (batch_size, num_max_boxes, num_anchors)
    assert is_in_topk.dtype == torch.float32

    # Check that each ground truth has at most 'topk' anchors selected
    assert (is_in_topk.sum(dim=-1) <= topk).all()


def test_get_final_assignments():
    # Constants for the test
    batch_size = 2
    num_max_boxes = 3
    num_anchors = 5
    num_classes = 80

    # Mock inputs
    gt_labels = torch.randint(0, num_classes, (batch_size, num_max_boxes, 1))
    gt_bboxes = torch.rand(batch_size, num_max_boxes, 4)
    assigned_gt_idx = torch.randint(0, num_max_boxes, (batch_size, num_anchors))
    mask_pos_sum = torch.randint(0, 2, (batch_size, num_anchors))

    # Initialize the TaskAlignedAssigner
    assigner = TaskAlignedAssigner(n_classes=num_classes, topk=13)
    assigner.bs = batch_size  # Set batch size
    assigner.n_max_boxes = gt_bboxes.size(1)

    # Call the method
    assigned_labels, assigned_bboxes, assigned_scores = assigner._get_final_assignments(
        gt_labels, gt_bboxes, assigned_gt_idx, mask_pos_sum
    )

    # Assert the expected outcomes
    assert assigned_labels.shape == (batch_size, num_anchors)
    assert assigned_bboxes.shape == (batch_size, num_anchors, 4)
    assert assigned_scores.shape == (batch_size, num_anchors, num_classes)
    assert (assigned_labels >= 0).all() and (assigned_labels <= num_classes).all()
