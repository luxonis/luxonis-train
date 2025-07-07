from typing import Literal

import pytest
import torch
from torch import Tensor

from luxonis_train.utils.boundingbox import (
    IoUType,
    anchors_for_fpn_features,
    bbox2dist,
    bbox_iou,
    compute_iou_loss,
    dist2bbox,
)


def generate_random_bboxes(
    n_bboxes: int,
    max_width: int,
    max_height: int,
    format: Literal["xyxy", "xywh", "cxcywh"],
):
    x1y1 = torch.rand(n_bboxes, 2) * torch.tensor(
        [max_width - 1, max_height - 1]
    )

    wh = (
        torch.rand(n_bboxes, 2)
        * (torch.tensor([max_width, max_height]) - 1 - x1y1)
        + 1
    )

    if format == "xyxy":
        x2y2 = x1y1 + wh
        bboxes = torch.cat((x1y1, x2y2), dim=1)
    elif format == "xywh":
        bboxes = torch.cat((x1y1, wh), dim=1)
    elif format == "cxcywh":
        cxcy = x1y1 + wh / 2
        bboxes = torch.cat((cxcy, wh), dim=1)

    return bboxes


def test_dist2bbox():
    distance = torch.rand(10, 4)
    anchor_points = torch.rand(10, 2)
    bbox = dist2bbox(distance, anchor_points)

    assert bbox.shape == distance.shape
    with pytest.raises(ValueError, match="'invalid'"):
        dist2bbox(distance, anchor_points, out_format="invalid")  # type: ignore


def test_bbox2dist():
    bbox = torch.rand(10, 4)
    anchor_points = torch.rand(10, 2)
    reg_max = 10.0

    distance = bbox2dist(bbox, anchor_points, reg_max)

    assert distance.shape == bbox.shape


@pytest.mark.parametrize("iou_type", ["none", "giou", "diou", "ciou", "siou"])
@pytest.mark.parametrize("format", ["xyxy", "xywh", "cxcywh"])
def test_bbox_iou(
    iou_type: IoUType, format: Literal["xyxy", "xywh", "cxcywh"]
):
    bbox1 = generate_random_bboxes(5, 640, 640, format)
    if iou_type == "siou":
        bbox2 = generate_random_bboxes(5, 640, 640, format)
    else:
        bbox2 = generate_random_bboxes(8, 640, 640, format)

    iou = bbox_iou(bbox1, bbox2, bbox_format=format, iou_type=iou_type)

    assert iou.shape == (bbox1.shape[0], bbox2.shape[0])
    min = 0 if iou_type == "none" else -1.5
    assert iou.min() >= min
    assert iou.max() <= 1

    if iou_type == "none":
        with pytest.raises(ValueError, match="'invalid' not supported"):
            bbox_iou(bbox1, bbox2, iou_type="invalid")  # type: ignore


def test_compute_iou_loss():
    pred_bboxes = generate_random_bboxes(8, 640, 640, "xyxy")
    target_bboxes = generate_random_bboxes(8, 640, 640, "xyxy")

    loss_iou, iou = compute_iou_loss(
        pred_bboxes, target_bboxes, iou_type="giou"
    )

    assert isinstance(loss_iou, Tensor)
    assert isinstance(iou, Tensor)
    assert iou.min() >= 0
    assert iou.max() <= 1


def test_anchors_for_fpn_features():
    features = [torch.rand(1, 256, 14, 14), torch.rand(1, 256, 28, 28)]
    strides = torch.tensor([8, 16])

    (anchors, anchor_points, n_anchors_list, stride_tensor) = (
        anchors_for_fpn_features(features, strides)
    )

    assert isinstance(anchors, Tensor)
    assert isinstance(anchor_points, Tensor)
    assert isinstance(n_anchors_list, list)
    assert isinstance(stride_tensor, Tensor)
    assert len(n_anchors_list) == len(features)
