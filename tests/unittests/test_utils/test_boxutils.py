from typing import Literal

import pytest
import torch
from torch import Tensor

from luxonis_train.utils.boundingbox import (
    IoUType,
    anchors_for_fpn_features,
    apply_bounding_box_to_masks,
    bbox2dist,
    bbox_iou,
    compute_iou_loss,
    dist2bbox,
    keypoints_to_bboxes,
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

    with pytest.raises(NotImplementedError, match="target_scores"):
        compute_iou_loss(
            pred_bboxes,
            target_bboxes,
            iou_type="giou",
            reduction="sum",
        )

    pred_bboxes_batch = pred_bboxes.unsqueeze(0)
    target_bboxes_batch = target_bboxes.unsqueeze(0)
    target_scores = torch.ones(1, 8, 1)
    mask_positive = torch.ones(1, 8, dtype=torch.bool)
    loss_sum, _ = compute_iou_loss(
        pred_bboxes_batch,
        target_bboxes_batch,
        target_scores=target_scores,
        mask_positive=mask_positive,
        iou_type="giou",
        reduction="sum",
    )
    assert isinstance(loss_sum, Tensor)

    with pytest.raises(ValueError, match="Unknown reduction"):
        compute_iou_loss(
            pred_bboxes_batch,
            target_bboxes_batch,
            mask_positive=mask_positive,
            iou_type="giou",
            reduction="invalid",  # type: ignore
        )

    loss_zero, iou_zero = compute_iou_loss(
        pred_bboxes.unsqueeze(0),
        target_bboxes.unsqueeze(0),
        mask_positive=torch.zeros(1, 8, dtype=torch.bool),
    )
    assert loss_zero.item() == 0
    assert iou_zero.tolist() == [0]


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

    anchors, anchor_points, _, _ = anchors_for_fpn_features(
        [torch.rand(1, 1, 1, 2)], torch.tensor([4]), multiply_with_stride=True
    )
    assert anchors.shape == (2, 4)
    assert anchor_points.tolist() == [[2.0, 2.0], [6.0, 2.0]]


def test_apply_bounding_box_to_masks():
    masks = torch.ones(2, 4, 4)
    boxes = torch.tensor([[1, 1, 3, 3], [0, 0, 2, 2]])

    cropped = apply_bounding_box_to_masks(masks, boxes)

    assert cropped[0].sum() == 4
    assert cropped[1].sum() == 4
    assert cropped[0, 0, 0] == 0
    assert cropped[1, 0, 0] == 1


def test_non_max_suppression_validation_and_empty():
    preds = torch.zeros(1, 1, 6)

    with pytest.raises(ValueError, match="Confidence threshold"):
        non_max_suppression(preds, n_classes=1, conf_thres=1.1)
    with pytest.raises(ValueError, match="IoU threshold"):
        non_max_suppression(preds, n_classes=1, iou_thres=-0.1)

    output = non_max_suppression(preds, n_classes=1, conf_thres=0.5)

    assert len(output) == 1
    assert output[0].shape == (0, 6)


def test_non_max_suppression_single_class_with_additional_data():
    preds = torch.tensor(
        [
            [
                [0.0, 0.0, 2.0, 2.0, 0.9, 0.1, 42.0],
                [10.0, 10.0, 12.0, 12.0, 0.8, 0.1, 43.0],
            ]
        ]
    )

    output = non_max_suppression(
        preds, n_classes=1, conf_thres=0.25, max_det=1
    )

    assert output[0].shape == (1, 7)
    assert output[0][0, 4].item() == pytest.approx(0.9)
    assert output[0][0, 5].item() == 0
    assert output[0][0, 6].item() == 42


def test_non_max_suppression_multi_label_keep_classes_and_xywh():
    preds = torch.tensor(
        [
            [
                [5.0, 5.0, 4.0, 4.0, 1.0, 0.4, 0.8],
                [20.0, 20.0, 2.0, 2.0, 1.0, 0.1, 0.95],
            ]
        ]
    )

    output = non_max_suppression(
        preds,
        n_classes=2,
        conf_thres=0.5,
        keep_classes=[1],
        multi_label=True,
        bbox_format="cxcywh",
        predicts_objectness=False,
    )

    assert output[0].shape == (2, 6)
    assert output[0][:, 5].tolist() == [1.0, 1.0]
    assert [3.0, 3.0, 7.0, 7.0] in output[0][:, :4].tolist()


def test_non_max_suppression_filters_all_classes():
    preds = torch.tensor([[[0.0, 0.0, 2.0, 2.0, 0.9, 0.9, 0.1]]])

    output = non_max_suppression(
        preds,
        n_classes=2,
        conf_thres=0.25,
        keep_classes=[1],
    )

    assert output[0].shape == (0, 7)


def test_keypoints_to_bboxes():
    empty = torch.empty((0, 1, 4))
    invisible = torch.tensor([[[5.0, 6.0, 0.1, 2.0]]])
    visible = torch.tensor(
        [
            [[1.0, 1.0, 0.5, 3.0]],
            [[9.0, 8.0, 0.9, 4.0]],
        ]
    )

    empty_boxes, invisible_boxes, visible_boxes = keypoints_to_bboxes(
        [empty, invisible, visible],
        img_height=10,
        img_width=10,
        box_width=4,
        visibility_threshold=0.5,
    )

    assert empty_boxes.shape == (0, 6)
    assert invisible_boxes.shape == (0, 6)
    torch.testing.assert_close(
        visible_boxes,
        torch.tensor(
            [
                [0.0, 0.0, 3.0, 3.0, 0.5, 3.0],
                [7.0, 6.0, 10.0, 10.0, 0.9, 4.0],
            ]
        ),
    )
