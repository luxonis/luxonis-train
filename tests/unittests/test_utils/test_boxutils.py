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
    non_max_suppression,
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


@pytest.mark.parametrize(
    ("conf_thres", "iou_thres", "message"),
    [(1.5, 0.45, "Confidence threshold"), (0.25, 1.5, "IoU threshold")],
)
def test_non_max_suppression_rejects_out_of_range_thresholds(
    conf_thres: float, iou_thres: float, message: str
):
    preds = torch.zeros(1, 2, 7)
    with pytest.raises(ValueError, match=message):
        non_max_suppression(
            preds, 2, conf_thres=conf_thres, iou_thres=iou_thres
        )


def test_non_max_suppression_multi_label_repeats_a_box_per_class():
    preds = torch.tensor(
        [
            [
                [10.0, 10.0, 20.0, 20.0, 1.0, 0.9, 0.8],
                [50.0, 50.0, 60.0, 60.0, 1.0, 0.1, 0.9],
            ]
        ]
    )
    out = non_max_suppression(preds, 2, conf_thres=0.5, multi_label=True)[0]

    # The first box clears the threshold on both classes, so it repeats.
    assert sorted(
        (int(label), round(conf, 2)) for *_, conf, label in out.tolist()
    ) == [(0, 0.9), (1, 0.8), (1, 0.9)]
    assert out[out[:, 5] == 0, :4].tolist() == [[10.0, 10.0, 20.0, 20.0]]


def test_non_max_suppression_multi_label_carries_additional_data():
    preds = torch.tensor(
        [
            [
                [10.0, 10.0, 20.0, 20.0, 1.0, 0.9, 0.8, 7.0],
                [50.0, 50.0, 60.0, 60.0, 1.0, 0.1, 0.9, 8.0],
            ]
        ]
    )
    out = non_max_suppression(preds, 2, conf_thres=0.5, multi_label=True)[0]

    assert out.shape == (3, 7)
    # Both rows of the repeated box keep that box's trailing column.
    assert sorted(out[:, 6].tolist()) == [7.0, 7.0, 8.0]


def test_non_max_suppression_single_class_copies_objectness():
    preds = torch.tensor([[[10.0, 10.0, 20.0, 20.0, 0.9, 0.2]]])
    out = non_max_suppression(preds, 1, conf_thres=0.5)[0]
    assert out.shape[0] == 1
    assert out[0, 4].item() == pytest.approx(0.9)


def test_non_max_suppression_filters_to_kept_classes():
    preds = torch.tensor(
        [
            [
                [10.0, 10.0, 20.0, 20.0, 0.9, 0.9, 0.1],
                [50.0, 50.0, 60.0, 60.0, 0.9, 0.1, 0.9],
            ]
        ]
    )
    kept = non_max_suppression(preds, 2, conf_thres=0.3, keep_classes=[1])[0]
    assert kept[:, 5].tolist() == [1.0]
    dropped = non_max_suppression(preds, 2, conf_thres=0.3, keep_classes=[7])[
        0
    ]
    assert dropped.shape[0] == 0


def test_compute_iou_loss_rejects_unknown_reduction():
    bboxes = torch.tensor([[10.0, 10.0, 20.0, 20.0]])
    with pytest.raises(ValueError, match="Unknown reduction type"):
        compute_iou_loss(
            bboxes,
            bboxes,
            reduction="bogus",  # type: ignore
        )


def test_non_max_suppression_converts_bbox_format():
    preds = torch.tensor([[[15.0, 15.0, 10.0, 10.0, 0.9, 0.9]]])
    out = non_max_suppression(preds, 1, conf_thres=0.5, bbox_format="cxcywh")[
        0
    ]
    assert out[0, :4].tolist() == [10.0, 10.0, 20.0, 20.0]
