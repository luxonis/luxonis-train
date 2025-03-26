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

    @type distance: Tensor
    @param distance: Distance predictions
    @type anchor_points: Tensor
    @param anchor_points: Head's anchor points
    @type out_format: BBoxFormatType
    @param out_format: BBox output format. Defaults to "xyxy".
    @rtype: Tensor
    @param dim: Dimension to split distance tensor. Defaults to -1.
    @rtype: Tensor
    @return: BBoxes in correct format
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

    @type bbox: Tensor
    @param bbox: Bboxes in "xyxy" format
    @type anchor_points: Tensor
    @param anchor_points: Head's anchor points
    @type reg_max: float
    @param reg_max: Maximum regression distances
    @rtype: Tensor
    @return: BBoxes in distance(ltrb) format
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
    """Computes IoU between two sets of bounding boxes.

    @type bbox1: Tensor
    @param bbox1: First set of bboxes [N, 4].
    @type bbox2: Tensor
    @param bbox2: Second set of bboxes [M, 4].
    @type bbox_format: BBoxFormatType
    @param bbox_format: Input bounding box format. Defaults to C{"xyxy"}.
    @type iou_type: Literal["none", "giou", "diou", "ciou", "siou"]
    @param iou_type: IoU type. Defaults to "none".
        Possible values are:
            - "none": standard IoU
            - "giou": Generalized IoU
            - "diou": Distance IoU
            - "ciou": Complete IoU. Introduced in U{
                Enhancing Geometric Factors in Model Learning and
                Inference for Object Detection and Instance
                Segmentation<https://arxiv.org/pdf/2005.03572.pdf>}.
                Implementation adapted from torchvision C{complete_box_iou}
                with improved stability.
            - "siou": Soft IoU. Introduced in U{
                SIoU Loss: More Powerful Learning for Bounding Box
                Regression<https://arxiv.org/pdf/2205.12740.pdf>}.
    @type element_wise: bool
    @param element_wise: If True returns element wise IoUs. Defaults to False.
    @rtype: Tensor
    @return: IoU between bbox1 and bbox2. If element_wise is True returns [N, M] tensor,
        otherwise returns [N] tensor.
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
    """Non-maximum suppression on model's predictions to keep only best
    instances.

    @type preds: Tensor
    @param preds: Model's prediction tensor of shape [bs, N, M].
    @type n_classes: int
    @param n_classes: Number of model's classes.
    @type conf_thres: float
    @param conf_thres: Boxes with confidence higher than this will be kept. Defaults to
        0.25.
    @type iou_thres: float
    @param iou_thres: Boxes with IoU higher than this will be discarded. Defaults to
        0.45.
    @type keep_classes: list[int] | None
    @param keep_classes: Subset of classes to keep, if None then keep all of them.
        Defaults to None.
    @type agnostic: bool
    @param agnostic: Whether perform NMS per class or treat all classes the same.
        Defaults to False.
    @type multi_label: bool
    @param multi_label: Whether one prediction can have multiple labels. Defaults to
        False.
    @type bbox_format: BBoxFormatType
    @param bbox_format: Input bbox format. Defaults to "xyxy".
    @type max_det: int
    @param max_det: Number of maximum output detections. Defaults to 300.
    @type predicts_objectness: bool
    @param predicts_objectness: Whether head predicts objectness confidence. Defaults to
        True.
    @rtype: list[Tensor]
    @return: list of kept detections for each image, boxes in "xyxy" format. Tensors
        with shape [n_kept, M]
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
    """Generates anchor boxes, points and strides based on FPN feature
    shapes and strides.

    @type features: list[Tensor]
    @param features: List of FPN features.
    @type strides: Tensor
    @param strides: Strides of FPN features.
    @type grid_cell_size: float
    @param grid_cell_size: Cell size in respect to input image size.
        Defaults to 5.0.
    @type grid_cell_offset: float
    @param grid_cell_offset: Percent grid cell center's offset. Defaults
        to 0.5.
    @type multiply_with_stride: bool
    @param multiply_with_stride: Whether to multiply per FPN values with
        its stride. Defaults to False.
    @rtype: tuple[Tensor, Tensor, list[int], Tensor]
    @return: BBox anchors, center anchors, number of anchors, strides
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
    """Crops the given masks to the regions specified by the
    corresponding bounding boxes.

    @type masks: Tensor
    @param masks: Masks tensor of shape [n, h, w].
    @type bounding_boxes: Tensor
    @param bounding_boxes: Bounding boxes tensor of shape [n, 4].
    @rtype: Tensor
    @return: Cropped masks tensor of shape [n, h, w].
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
    """Computes an IoU loss between 2 sets of bounding boxes.

    @type pred_bboxes: Tensor
    @param pred_bboxes: Predicted bounding boxes.
    @type target_bboxes: Tensor
    @param target_bboxes: Target bounding boxes.
    @type target_scores: Tensor | None
    @param target_scores: Target scores. Defaults to None.
    @type mask_positive: Tensor | None
    @param mask_positive: Mask for positive samples. Defaults to None.
    @type iou_type: L{IoUType}
    @param iou_type: IoU type. Defaults to "giou".
    @type bbox_format: L{BBoxFormatType}
    @param bbox_format: BBox format. Defaults to "xyxy".
    @type reduction: Literal["sum", "mean"]
    @param reduction: Reduction type. Defaults to "mean".
    @rtype: tuple[Tensor, Tensor]
    @return: IoU loss and IoU values.
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
    """Convert keypoints to bounding boxes in xyxy format with cls_id
    and score, filtering low-visibility keypoints.

    @type keypoints: list[Tensor]
    @param keypoints: List of tensors of keypoints with shape [N, 1, 4]
        (x, y, v, cls_id).
    @type img_height: int
    @param img_height: Height of the image.
    @type img_width: int
    @param img_width: Width of the image.
    @type box_width: int
    @param box_width: Width of the bounding box in pixels. Defaults to
        2.
    @type visibility_threshold: float
    @param visibility_threshold: Minimum visibility score to include a
        keypoint. Defaults to 0.5.
    @rtype: list[Tensor]
    @return: List of tensors of bounding boxes with shape [N, 6] (x_min,
        y_min, x_max, y_max, score, cls_id).
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
