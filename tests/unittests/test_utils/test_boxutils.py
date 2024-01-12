import torch

from luxonis_train.utils.boxutils import (
    anchors_for_fpn_features,
    bbox2dist,
    bbox_iou,
    compute_iou_loss,
    dist2bbox,
    process_bbox_predictions,
    process_keypoints_predictions,
)


def generate_random_bboxes(num_bboxes, max_width, max_height, format="xyxy"):
    # Generate top-left corners (x1, y1)
    x1y1 = torch.rand(num_bboxes, 2) * torch.tensor([max_width - 1, max_height - 1])

    # Generate widths and heights ensuring x2 > x1 and y2 > y1
    wh = (
        torch.rand(num_bboxes, 2) * (torch.tensor([max_width, max_height]) - 1 - x1y1)
        + 1
    )

    if format == "xyxy":
        # Calculate bottom-right corners (x2, y2) for xyxy format
        x2y2 = x1y1 + wh
        bboxes = torch.cat((x1y1, x2y2), dim=1)
    elif format == "xywh":
        # Use x1y1 as top-left corner and wh as width and height for xywh format
        bboxes = torch.cat((x1y1, wh), dim=1)
    elif format == "cxcywh":
        # Calculate center coordinates and use wh as width and height for cxcywh format
        cxcy = x1y1 + wh / 2
        bboxes = torch.cat((cxcy, wh), dim=1)
    else:
        raise ValueError("Unsupported format. Choose from 'xyxy', 'xywh', 'cxcywh'.")

    return bboxes


def test_dist2bbox():
    distance = torch.rand(10, 4)
    anchor_points = torch.rand(10, 2)
    bbox = dist2bbox(distance, anchor_points)

    assert bbox.shape == distance.shape


def test_bbox2dist():
    bbox = torch.rand(10, 4)
    anchor_points = torch.rand(10, 2)
    reg_max = 10.0

    distance = bbox2dist(bbox, anchor_points, reg_max)

    assert distance.shape == bbox.shape


def test_bbox_iou():
    for format in ["xyxy", "cxcywh", "xywh"]:
        bbox1 = generate_random_bboxes(5, 640, 640, format)
        bbox2 = generate_random_bboxes(8, 640, 640, format)

        iou = bbox_iou(bbox1, bbox2)

        assert iou.shape == (5, 8)
        assert iou.min() >= 0 and iou.max() <= 1


def test_compute_iou_loss():
    pred_bboxes = generate_random_bboxes(8, 640, 640, "xyxy")
    target_bboxes = generate_random_bboxes(8, 640, 640, "xyxy")

    loss_iou, iou = compute_iou_loss(pred_bboxes, target_bboxes, iou_type="giou")

    assert isinstance(loss_iou, torch.Tensor)
    assert isinstance(iou, torch.Tensor)
    assert 0 <= iou.min() and iou.max() <= 1


def test_process_bbox_predictions():
    bbox = generate_random_bboxes(10, 64, 64, "xywh")
    data = torch.rand(10, 4)
    prediction = torch.concat([bbox, data], dim=-1)
    anchor = torch.rand(10, 2)

    out_bbox_xy, out_bbox_wh, out_bbox_tail = process_bbox_predictions(
        prediction, anchor
    )

    assert out_bbox_xy.shape == (10, 2)
    assert out_bbox_wh.shape == (10, 2)
    assert out_bbox_tail.shape == (10, 4)


def test_process_keypoints_predictions():
    keypoints = torch.rand(10, 15)  # 5 keypoints * 3 (x, y, visibility)

    x, y, visibility = process_keypoints_predictions(keypoints)

    assert x.shape == y.shape == visibility.shape == (10, 5)


def test_anchors_for_fpn_features():
    features = [torch.rand(1, 256, 14, 14), torch.rand(1, 256, 28, 28)]
    strides = torch.tensor([8, 16])

    anchors, anchor_points, n_anchors_list, stride_tensor = anchors_for_fpn_features(
        features, strides
    )

    assert isinstance(anchors, torch.Tensor)
    assert isinstance(anchor_points, torch.Tensor)
    assert isinstance(n_anchors_list, list)
    assert isinstance(stride_tensor, torch.Tensor)
    assert len(n_anchors_list) == len(features)
