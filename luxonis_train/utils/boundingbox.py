import math
from typing import Literal, TypeAlias

import torch
from torch import Tensor
from torchvision.ops import (
    batched_nms,
    box_convert,
    box_iou,
    distance_box_iou,
    generalized_box_iou,
)

IoUType: TypeAlias = Literal["none", "giou", "diou", "ciou", "siou"]
BBoxFormatType: TypeAlias = Literal["xyxy", "xywh", "cxcywh"]


def dist2bbox(
    distance: Tensor,
    anchor_points: Tensor,
    out_format: BBoxFormatType = "xyxy",
    dim: int = -1,
) -> Tensor:
    """Transform distance (ltrb) to box ("xyxy", "xywh" or "cxcywh").

    Args:
        distance (Tensor): Distance predictions.
        anchor_points (Tensor): Head anchor points.
        out_format (BBoxFormatType): BBox output format. Defaults to
            ``"xyxy"``.
        dim (int): Dimension to split the distance tensor on. Defaults to
            ``-1``.

    Returns:
        Tensor: Bounding boxes in `out_format`.

    Raises:
        ValueError: If `out_format` is not supported.

    """
    lt, rb = torch.split(distance, 2, dim=dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    bbox = torch.cat([x1y1, x2y2], dim=dim)
    if out_format in {"xyxy", "xywh", "cxcywh"}:
        bbox = box_convert(bbox, in_fmt="xyxy", out_fmt=out_format)
    else:
        raise ValueError(f"Out format '{out_format}' for bbox not supported")
    return bbox


def bbox2dist(bbox: Tensor, anchor_points: Tensor, reg_max: float) -> Tensor:
    """Transform bbox(xyxy) to distance(ltrb).

    Args:
        bbox (Tensor): Bounding boxes in ``"xyxy"`` format.
        anchor_points (Tensor): Head anchor points.
        reg_max (float): Maximum regression distance.

    Returns:
        Tensor: Bounding boxes in distance ``ltrb`` format.

    """
    x1y1, x2y2 = torch.split(bbox, 2, -1)
    lt = anchor_points - x1y1
    rb = x2y2 - anchor_points
    return torch.cat([lt, rb], -1).clip(0, reg_max - 0.01)


# CLEAN:
def bbox_iou(
    bbox1: Tensor,
    bbox2: Tensor,
    bbox_format: BBoxFormatType = "xyxy",
    iou_type: IoUType = "none",
    element_wise: bool = False,
) -> Tensor:
    """Compute IoU between two sets of bounding boxes.

    Args:
        bbox1 (Tensor): First set of bounding boxes with shape ``[N, 4]``.
        bbox2 (Tensor): Second set of bounding boxes with shape ``[M, 4]``.
        bbox_format (BBoxFormatType): Input bounding box format. Defaults to
            ``"xyxy"``.
        iou_type (IoUType): IoU type. Defaults to ``"none"``. Supported values
            are ``"none"`` for standard IoU, ``"giou"`` for Generalized IoU,
            ``"diou"`` for Distance IoU, ``"ciou"`` for Complete IoU from
            `Enhancing Geometric Factors in Model Learning and Inference for
            Object Detection and Instance Segmentation
            <https://arxiv.org/pdf/2005.03572.pdf>`_, and ``"siou"`` for Soft
            IoU from `SIoU Loss: More Powerful Learning for Bounding Box
            Regression <https://arxiv.org/pdf/2205.12740.pdf>`_. The CIoU
            implementation is adapted from torchvision
            ``complete_box_iou`` with improved stability.
        element_wise (bool): If ``True``, return element-wise IoUs. Defaults to
            ``False``.

    Returns:
        Tensor: IoU between `bbox1` and `bbox2`. When `element_wise` is
        ``True``, returns shape ``[N]``; otherwise returns shape ``[N, M]``.

    Raises:
        ValueError: If `iou_type` is not supported.

    """
    if bbox_format != "xyxy":
        bbox1 = box_convert(bbox1, in_fmt=bbox_format, out_fmt="xyxy")
        bbox2 = box_convert(bbox2, in_fmt=bbox_format, out_fmt="xyxy")

    if iou_type == "none":
        iou = box_iou(bbox1, bbox2)
    elif iou_type == "giou":
        iou = generalized_box_iou(bbox1, bbox2)
    elif iou_type == "diou":
        iou = distance_box_iou(bbox1, bbox2)
    elif iou_type == "ciou":
        eps = 1e-7

        iou = bbox_iou(bbox1, bbox2, iou_type="none")
        diou = bbox_iou(bbox1, bbox2, iou_type="diou")

        w1 = bbox1[:, None, 2] - bbox1[:, None, 0]
        h1 = bbox1[:, None, 3] - bbox1[:, None, 1] + eps
        w2 = bbox2[:, 2] - bbox2[:, 0]
        h2 = bbox2[:, 3] - bbox2[:, 1] + eps

        v = (4 / (torch.pi**2)) * torch.pow(
            torch.atan(w1 / h1) - torch.atan(w2 / h2), 2
        )
        with torch.no_grad():
            alpha = v / (1 - iou + v + eps)
        iou = diou - alpha * v

    elif iou_type == "siou":
        eps = 1e-7
        bbox1_xywh = box_convert(bbox1, in_fmt="xyxy", out_fmt="xywh")
        w1, h1 = bbox1_xywh[:, 2], bbox1_xywh[:, 3]
        bbox2_xywh = box_convert(bbox2, in_fmt="xyxy", out_fmt="xywh")
        w2, h2 = bbox2_xywh[:, 2], bbox2_xywh[:, 3]

        # enclose area
        enclose_x1y1 = torch.min(bbox1[:, None, :2], bbox2[:, :2])
        enclose_x2y2 = torch.max(bbox1[:, None, 2:], bbox2[:, 2:])
        enclose_wh = (enclose_x2y2 - enclose_x1y1).clamp(min=eps)
        cw = enclose_wh[..., 0]
        ch = enclose_wh[..., 1]

        # angle cost
        s_cw = (
            bbox2[:, None, 0] + bbox2[:, None, 2] - bbox1[:, 0] - bbox1[:, 2]
        ) * 0.5 + eps
        s_ch = (
            bbox2[:, None, 1] + bbox2[:, None, 3] - bbox1[:, 1] - bbox1[:, 3]
        ) * 0.5 + eps

        sigma = torch.pow(s_cw**2 + s_ch**2, 0.5)

        sin_alpha_1 = torch.abs(s_cw) / sigma
        sin_alpha_2 = torch.abs(s_ch) / sigma
        threshold = pow(2, 0.5) / 2
        sin_alpha = torch.where(
            sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1
        )
        angle_cost = torch.cos(torch.arcsin(sin_alpha) * 2 - math.pi / 2)

        # distance cost
        rho_x = (s_cw / cw) ** 2
        rho_y = (s_ch / ch) ** 2
        gamma = angle_cost - 2
        distance_cost = 2 - torch.exp(gamma * rho_x) - torch.exp(gamma * rho_y)

        # shape cost
        omega_w = torch.abs(w1 - w2) / torch.max(w1, w2)
        omega_h = torch.abs(h1 - h2) / torch.max(h1, h2)
        shape_cost = torch.pow(1 - torch.exp(-1 * omega_w), 4) + torch.pow(
            1 - torch.exp(-1 * omega_h), 4
        )

        iou = box_iou(bbox1, bbox2) - 0.5 * (distance_cost + shape_cost)
    else:
        raise ValueError(f"IoU type '{iou_type}' not supported.")

    iou = torch.nan_to_num(iou, 0)

    if element_wise:
        return iou.diag()
    return iou


def non_max_suppression(
    preds: Tensor,
    n_classes: int,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    keep_classes: list[int] | None = None,
    agnostic: bool = False,
    multi_label: bool = False,
    bbox_format: BBoxFormatType = "xyxy",
    max_det: int = 300,
    predicts_objectness: bool = True,
) -> list[Tensor]:
    """Run non-maximum suppression on model predictions.

    Args:
        preds (Tensor): Model prediction tensor with shape ``[bs, N, M]``.
        n_classes (int): Number of model classes.
        conf_thres (float): Boxes with confidence higher than this value are
            kept. Defaults to ``0.25``.
        iou_thres (float): Boxes with IoU higher than this value are
            discarded. Defaults to ``0.45``.
        keep_classes (list[int] | None): Subset of classes to keep. If
            ``None``, all classes are kept. Defaults to ``None``.
        agnostic (bool): Whether to treat all classes the same during NMS.
            Defaults to ``False``.
        multi_label (bool): Whether one prediction can have multiple labels.
            Defaults to ``False``.
        bbox_format (BBoxFormatType): Input bounding box format. Defaults to
            ``"xyxy"``.
        max_det (int): Maximum number of output detections. Defaults to
            ``300``.
        predicts_objectness (bool): Whether the head predicts objectness
            confidence. Defaults to ``True``.

    Returns:
        list[Tensor]: Kept detections for each image, with boxes in ``"xyxy"``
        format and tensors shaped ``[n_kept, M]``.

    Raises:
        ValueError: If `conf_thres` or `iou_thres` is outside ``[0, 1]``.

    """
    if not (0 <= conf_thres <= 1):
        raise ValueError(
            f"Confidence threshold must be in range [0,1] but set to {conf_thres}."
        )
    if not (0 <= iou_thres <= 1):
        raise ValueError(
            f"IoU threshold must be in range [0,1] but set to {iou_thres}."
        )

    multi_label &= n_classes > 1

    # If any data after bboxes are present.
    has_additional = preds.size(-1) > (4 + 1 + n_classes)

    candidate_mask = preds[..., 4] > conf_thres
    if not predicts_objectness:
        candidate_mask = torch.logical_and(
            candidate_mask,
            torch.max(preds[..., 5 : 5 + n_classes], dim=-1)[0] > conf_thres,
        )

    output = [
        torch.zeros((0, preds.size(-1)), device=preds.device)
    ] * preds.size(0)

    for i, x in enumerate(preds):
        curr_out = x[candidate_mask[i]]

        if curr_out.size(0) == 0:
            continue

        if predicts_objectness:
            if n_classes == 1:
                curr_out[:, 5 : 5 + n_classes] = curr_out[:, 4:5]
            else:
                curr_out[:, 5 : 5 + n_classes] *= curr_out[:, 4:5]
        else:
            curr_out[:, 5 : 5 + n_classes] *= curr_out[:, 4:5]

        bboxes = curr_out[:, :4]
        keep_mask = torch.zeros(bboxes.size(0)).bool()
        if bbox_format != "xyxy":
            bboxes = box_convert(bboxes, in_fmt=bbox_format, out_fmt="xyxy")

        if multi_label:
            box_idx, class_idx = (
                (curr_out[:, 5 : 5 + n_classes] > conf_thres)
                .nonzero(as_tuple=False)
                .T
            )
            keep_mask[box_idx] = True
            curr_out = torch.cat(
                (
                    bboxes[keep_mask],
                    curr_out[keep_mask, class_idx + 5, None],
                    class_idx[:, None].float(),
                ),
                1,
            )
        else:
            conf, class_idx = curr_out[:, 5 : 5 + n_classes].max(
                1, keepdim=True
            )
            keep_mask[conf.view(-1) > conf_thres] = True
            curr_out = torch.cat((bboxes, conf, class_idx.float()), 1)[
                keep_mask
            ]

        if has_additional:
            curr_out = torch.hstack(
                [curr_out, x[candidate_mask[i]][keep_mask, 5 + n_classes :]]
            )

        if keep_classes is not None:
            curr_out = curr_out[
                (
                    curr_out[:, 5:6]
                    == torch.tensor(keep_classes, device=curr_out.device)
                ).any(1)
            ]

        if not curr_out.size(0):
            continue

        keep_indices = batched_nms(
            boxes=curr_out[:, :4],
            scores=curr_out[:, 4],
            iou_threshold=iou_thres,
            idxs=curr_out[:, 5].int() * (0 if agnostic else 1),
        )
        keep_indices = keep_indices[:max_det]

        output[i] = curr_out[keep_indices]

    return output


def anchors_for_fpn_features(
    features: list[Tensor],
    strides: Tensor,
    grid_cell_size: float = 5.0,
    grid_cell_offset: float = 0.5,
    multiply_with_stride: bool = False,
) -> tuple[Tensor, Tensor, list[int], Tensor]:
    """Generate anchor boxes, points, and strides for FPN features.

    Args:
        features (list[Tensor]): FPN feature tensors.
        strides (Tensor): Strides of the FPN features.
        grid_cell_size (float): Cell size with respect to input image size.
            Defaults to ``5.0``.
        grid_cell_offset (float): Percent offset of the grid cell center.
            Defaults to ``0.5``.
        multiply_with_stride (bool): Whether to multiply per-FPN values with
            their stride. Defaults to ``False``.

    Returns:
        tuple[Tensor, Tensor, list[int], Tensor]: A tuple containing bounding
        box anchors, center anchors, number of anchors per feature map, and
        stride tensor.

    """
    anchors: list[Tensor] = []
    anchor_points: list[Tensor] = []
    n_anchors_list: list[int] = []
    stride_tensor: list[Tensor] = []
    # FIXME: strict=True
    for feature, stride in zip(features, strides, strict=False):
        _, _, h, w = feature.shape
        cell_half_size = grid_cell_size * stride * 0.5
        shift_x = torch.arange(end=w) + grid_cell_offset
        shift_y = torch.arange(end=h) + grid_cell_offset
        if multiply_with_stride:
            shift_x *= stride
            shift_y *= stride
        shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing="ij")

        anchor = (
            torch.stack(
                [
                    shift_x - cell_half_size,
                    shift_y - cell_half_size,
                    shift_x + cell_half_size,
                    shift_y + cell_half_size,
                ],
                dim=-1,
            )
            .reshape(-1, 4)
            .to(feature.dtype)
        )
        anchors.append(anchor)

        anchor_point = (
            torch.stack([shift_x, shift_y], dim=-1)
            .reshape(-1, 2)
            .to(feature.dtype)
        )
        anchor_points.append(anchor_point)

        curr_n_anchors = len(anchor)
        n_anchors_list.append(curr_n_anchors)
        stride_tensor.append(
            torch.full((curr_n_anchors, 1), stride.item(), dtype=feature.dtype)
        )

    device = features[0].device
    return (
        torch.cat(anchors).to(device),
        torch.cat(anchor_points).to(device),
        n_anchors_list,
        torch.cat(stride_tensor).to(device),
    )


def apply_bounding_box_to_masks(
    masks: Tensor, bounding_boxes: Tensor
) -> Tensor:
    """Crop masks to the regions specified by corresponding boxes.

    Args:
        masks (Tensor): Masks tensor with shape ``[n, h, w]``.
        bounding_boxes (Tensor): Bounding boxes tensor with shape ``[n, 4]``.

    Returns:
        Tensor: Cropped masks tensor with shape ``[n, h, w]``.

    """
    _, mask_height, mask_width = masks.shape
    left, top, right, bottom = torch.split(
        bounding_boxes[:, :, None], 1, dim=1
    )
    width_indices = torch.arange(
        mask_width, device=masks.device, dtype=left.dtype
    )[None, None, :]
    height_indices = torch.arange(
        mask_height, device=masks.device, dtype=left.dtype
    )[None, :, None]

    return masks * (
        (width_indices >= left)
        & (width_indices < right)
        & (height_indices >= top)
        & (height_indices < bottom)
    )


def compute_iou_loss(
    pred_bboxes: Tensor,
    target_bboxes: Tensor,
    target_scores: Tensor | None = None,
    mask_positive: Tensor | None = None,
    *,
    iou_type: IoUType = "giou",
    bbox_format: BBoxFormatType = "xyxy",
    reduction: Literal["sum", "mean"] = "mean",
) -> tuple[Tensor, Tensor]:
    """Compute an IoU loss between 2 sets of bounding boxes.

    Args:
        pred_bboxes (Tensor): Predicted bounding boxes.
        target_bboxes (Tensor): Target bounding boxes.
        target_scores (Tensor | None): Target scores. Defaults to ``None``.
        mask_positive (Tensor | None): Mask for positive samples. Defaults to
            ``None``.
        iou_type (IoUType): IoU type. Defaults to ``"giou"``.
        bbox_format (BBoxFormatType): Bounding box format. Defaults to
            ``"xyxy"``.
        reduction (Literal["sum", "mean"]): Reduction type. Defaults to
            ``"mean"``.

    Returns:
        tuple[Tensor, Tensor]: IoU loss and detached IoU values.

    Raises:
        NotImplementedError: If ``reduction="sum"`` is used without
            `target_scores`.
        ValueError: If `reduction` or `iou_type` is unsupported.

    """
    device = pred_bboxes.device
    target_bboxes = target_bboxes.to(device)
    if mask_positive is None or mask_positive.sum() > 0:
        if target_scores is not None:
            bbox_weight = torch.masked_select(
                target_scores.sum(-1),
                mask_positive
                if mask_positive is not None
                else torch.ones_like(target_scores.sum(-1)),
            ).unsqueeze(-1)
        else:
            bbox_weight = torch.tensor(1.0)

        if mask_positive is not None:
            bbox_mask = mask_positive.unsqueeze(-1).repeat([1, 1, 4])
        else:
            bbox_mask = torch.ones_like(pred_bboxes, dtype=torch.bool)

        pred_bboxes_pos = torch.masked_select(pred_bboxes, bbox_mask).reshape(
            [-1, 4]
        )
        target_bboxes_pos = torch.masked_select(
            target_bboxes, bbox_mask
        ).reshape([-1, 4])

        iou = bbox_iou(
            pred_bboxes_pos,
            target_bboxes_pos,
            iou_type=iou_type,
            bbox_format=bbox_format,
            element_wise=True,
        ).unsqueeze(-1)
        loss_iou = (1 - iou) * bbox_weight

        if reduction == "mean":
            loss_iou = loss_iou.mean()

        elif reduction == "sum":
            if target_scores is None:
                raise NotImplementedError(
                    "Sum reduction is not supported when `target_scores` is None"
                )
            loss_iou = loss_iou.sum()
            if target_scores.sum() > 1:
                loss_iou /= target_scores.sum()
        else:
            raise ValueError(f"Unknown reduction type `{reduction}`")
    else:
        loss_iou = torch.tensor(0.0).to(pred_bboxes.device)
        iou = torch.zeros([target_bboxes.shape[0]]).to(pred_bboxes.device)

    return loss_iou, iou.detach().clamp(0)


def keypoints_to_bboxes(
    keypoints: list[Tensor],
    img_height: int,
    img_width: int,
    box_width: int = 5,
    visibility_threshold: float = 0.5,
) -> list[Tensor]:
    """Convert keypoints to bounding boxes in ``xyxy`` format.

    Low-visibility keypoints are filtered out.

    Args:
        keypoints (list[Tensor]): Keypoint tensors with shape ``[N, 1, 4]`` in
            ``(x, y, v, cls_id)`` order.
        img_height (int): Image height.
        img_width (int): Image width.
        box_width (int): Bounding box width in pixels. Defaults to ``5``.
        visibility_threshold (float): Minimum visibility score required to
            include a keypoint. Defaults to ``0.5``.

    Returns:
        list[Tensor]: Bounding box tensors with shape ``[N, 6]`` in
        ``(x_min, y_min, x_max, y_max, score, cls_id)`` order.

    """
    half_box = box_width / 2
    bboxes_list = []

    for keypoints_per_image in keypoints:
        if keypoints_per_image.numel() == 0:
            bboxes_list.append(
                torch.zeros((0, 6), device=keypoints_per_image.device)
            )
            continue

        keypoints_per_image = keypoints_per_image.squeeze(1)

        visible_mask = keypoints_per_image[:, 2] >= visibility_threshold
        keypoints_per_image = keypoints_per_image[visible_mask]

        if keypoints_per_image.numel() == 0:
            bboxes_list.append(
                torch.zeros((0, 6), device=keypoints_per_image.device)
            )
            continue

        x_coords = keypoints_per_image[:, 0]
        y_coords = keypoints_per_image[:, 1]
        scores = keypoints_per_image[:, 2]
        cls_ids = keypoints_per_image[:, 3]

        x_min = (x_coords - half_box).clamp(min=0)
        y_min = (y_coords - half_box).clamp(min=0)
        x_max = (x_coords + half_box).clamp(max=img_width)
        y_max = (y_coords + half_box).clamp(max=img_height)
        bboxes = torch.stack(
            [x_min, y_min, x_max, y_max, scores, cls_ids], dim=-1
        )
        bboxes_list.append(bboxes)

    return bboxes_list
