"""Bounding box helpers for the detection heads, losses, and metrics.

The module holds:

- the conversions between box corners and distances from anchor points;
- the IoU, its variants, and an IoU loss;
- non-maximum suppression;
- the anchors of the feature maps of a feature pyramid;
- the removal of the mask pixels outside of boxes, and the conversion
  of keypoints to boxes.

`BBoxFormatType` names the box formats:

- ``"xyxy"``: the top-left corner and the bottom-right corner,
  ``(x1, y1, x2, y2)``.
- ``"xywh"``: the top-left corner, the width, and the height,
  ``(x, y, w, h)``.
- ``"cxcywh"``: the center, the width, and the height,
  ``(cx, cy, w, h)``.

`IoUType` names the IoU variants of `bbox_iou`: ``"none"``, ``"giou"``,
``"diou"``, ``"ciou"``, and ``"siou"``.

"""

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
    r"""Convert distances from anchor points to boxes.

    The distances ``(l, t, r, b)`` go from an anchor point
    :math:`(x, y)` to the left, top, right, and bottom side of a box.
    The function computes the corners :math:`(x - l, y - t)` and
    :math:`(x + r, y + b)`, and converts the box to ``out_format``.
    `bbox2dist` does the opposite conversion.

    Args:
        distance (``Tensor``): The distances ``(l, t, r, b)``, with size
            ``4`` on dimension ``dim``.
        anchor_points (``Tensor``): The anchor points ``(x, y)``. The
            shape must broadcast with one half of ``distance``, which has
            size ``2`` on dimension ``dim``.
        out_format (BBoxFormatType): The format of the returned boxes.
        dim (int): The dimension that holds the coordinates. Only the
            ``"xyxy"`` format supports a dimension other than the last
            one, because the format conversion reads the last dimension.

    Returns:
        ``Tensor``: The boxes in ``out_format``, with size ``4`` on
        dimension ``dim``.

    Raises:
        ValueError: When ``out_format`` is not ``"xyxy"``, ``"xywh"``, or
            ``"cxcywh"``.

    Example:
        >>> import torch
        >>> distance = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        >>> anchor_points = torch.tensor([[10.0, 10.0]])
        >>> dist2bbox(distance, anchor_points).tolist()
        [[9.0, 8.0, 13.0, 14.0]]
        >>> dist2bbox(distance, anchor_points, out_format="cxcywh").tolist()
        [[11.0, 11.0, 4.0, 6.0]]

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
    """Convert ``xyxy`` boxes to distances from anchor points.

    The function computes the distances ``(l, t, r, b)`` from each anchor
    point to the left, top, right, and bottom side of its box. It clips
    the distances to ``[0, reg_max - 0.01]``. A distance is ``0`` when
    the anchor point is outside of the box on that side. `dist2bbox`
    does the opposite conversion.

    Args:
        bbox (``Tensor``): The boxes in ``xyxy`` format, of shape
            ``[..., 4]``.
        anchor_points (``Tensor``): The anchor points ``(x, y)``, of shape
            ``[..., 2]``. The shape must broadcast with the shape of the
            box corners.
        reg_max (float): The limit of a distance. The largest returned
            distance is ``reg_max - 0.01``.

    Returns:
        ``Tensor``: The distances ``(l, t, r, b)``, of shape ``[..., 4]``.

    Example:
        >>> import torch
        >>> bbox = torch.tensor([[9.0, 8.0, 13.0, 14.0]])
        >>> anchor_points = torch.tensor([[10.0, 10.0]])
        >>> bbox2dist(bbox, anchor_points, reg_max=16).tolist()
        [[1.0, 2.0, 3.0, 4.0]]
        >>> distances = bbox2dist(bbox, anchor_points, reg_max=3)
        >>> [round(value, 2) for value in distances[0].tolist()]
        [1.0, 2.0, 2.99, 2.99]

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
    r"""Compute the IoU between two sets of bounding boxes.

    The function converts both sets to ``xyxy`` and computes the IoU of
    every box in ``bbox1`` with every box in ``bbox2``. :math:`A` and
    :math:`B` are two boxes, and :math:`C` is the smallest box that
    encloses both. ``iou_type`` selects the variant:

    - ``"none"``: the plain IoU,
      :math:`\text{IoU} = \frac{|A \cap B|}{|A \cup B|}`.
    - ``"giou"``: the generalized IoU,
      :math:`\text{IoU} - \frac{|C \setminus (A \cup B)|}{|C|}`.
    - ``"diou"``: the distance IoU,
      :math:`\text{IoU} - \frac{\rho^2}{c^2}`. :math:`\rho` is the
      distance between the box centers, and :math:`c` is the diagonal
      of :math:`C`.
    - ``"ciou"``: the complete IoU from `Enhancing Geometric Factors in
      Model Learning and Inference for Object Detection and Instance
      Segmentation <https://arxiv.org/pdf/2005.03572.pdf>`_,
      :math:`\text{DIoU} - \alpha v`.
    - ``"siou"``: the SIoU from `SIoU Loss: More Powerful Learning for
      Bounding Box Regression <https://arxiv.org/pdf/2205.12740.pdf>`_,
      :math:`\text{IoU} - \frac{\Delta + \Omega}{2}`. :math:`\Delta` is
      the distance cost, which includes the angle cost, and
      :math:`\Omega` is the shape cost.

    For ``"ciou"``, :math:`w` and :math:`h` are the width and the height
    of a box. The function adds :math:`10^{-7}` to the heights, and no
    gradient flows through :math:`\alpha`:

    .. math::

        v = \frac{4}{\pi^2} \left(\arctan\frac{w_A}{h_A}
        - \arctan\frac{w_B}{h_B}\right)^2

        \alpha = \frac{v}{1 - \text{IoU} + v + 10^{-7}}

    The function replaces a ``NaN`` result with ``0``.

    **Warning:** ``"siou"`` pairs ``bbox1[i]`` with ``bbox2[i]`` in some
    of its terms. It needs ``N`` equal to ``M``, and only the diagonal of
    its result is correct. Use it with ``element_wise=True``.

    Args:
        bbox1 (``Tensor``): The first set of boxes, of shape ``[N, 4]``.
        bbox2 (``Tensor``): The second set of boxes, of shape ``[M, 4]``.
        bbox_format (BBoxFormatType): The format of both sets of boxes.
        iou_type (IoUType): The IoU variant.
        element_wise (bool): Whether to return only the diagonal of the
            IoU matrix, the IoU of ``bbox1[i]`` and ``bbox2[i]``. The
            function computes the full matrix in both cases.

    Returns:
        ``Tensor``: The IoU matrix of shape ``[N, M]``. With
        ``element_wise``, its diagonal, of shape ``[min(N, M)]``.

    Raises:
        ValueError: When ``iou_type`` is not a supported variant.

    Example:
        >>> import torch
        >>> boxes = torch.tensor([[0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 2.0, 1.0]])
        >>> target = torch.tensor([[0.0, 0.0, 2.0, 2.0]])
        >>> bbox_iou(boxes, target).tolist()
        [[1.0], [0.5]]
        >>> bbox_iou(boxes, boxes, element_wise=True).tolist()
        [1.0, 1.0]

        The generalized IoU of two separate boxes is below ``0``:

        >>> left = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
        >>> right = torch.tensor([[2.0, 0.0, 3.0, 1.0]])
        >>> round(bbox_iou(left, right, iou_type="giou").item(), 4)
        -0.3333

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
    """Run non-maximum suppression on the box predictions of a batch.

    A row of ``preds`` holds a box, a confidence, the class scores, and
    optional extra values, such as keypoints or mask coefficients. For
    each image, the function does these steps:

    - It keeps the rows with a confidence above ``conf_thres``. When
      ``predicts_objectness`` is ``False``, the highest class score must
      also be above ``conf_thres``.
    - It multiplies the class scores by the confidence. When
      ``predicts_objectness`` is ``True`` and ``n_classes`` is ``1``, it
      replaces the class score with the confidence.
    - It gives each row the class with the highest score, and keeps the
      rows with that score above ``conf_thres``. With ``multi_label``,
      a row gets one copy for each class with a score above
      ``conf_thres``.
    - It drops the rows with a class that is not in ``keep_classes``.
    - It runs NMS for each class, or across all classes when
      ``agnostic`` is ``True``. It keeps at most ``max_det`` rows, the
      rows with the highest scores.

    Args:
        preds (``Tensor``): The predictions, of shape ``[B, N, M]``. The
            columns of a row are the box in ``bbox_format``, the
            confidence, ``n_classes`` class scores, and
            ``E = M - 5 - n_classes`` extra values.
        n_classes (int): The number of class score columns in ``preds``.
        conf_thres (float): The score threshold, in ``[0, 1]``. A kept
            score is strictly above it.
        iou_thres (float): The IoU threshold of NMS, in ``[0, 1]``. NMS
            drops a box when its IoU with a box of higher score is above
            this value.
        keep_classes (list[int] | None): The indices of the classes to
            keep. ``None`` keeps all classes.
        agnostic (bool): Whether NMS compares the boxes of different
            classes.
        multi_label (bool): Whether a box can get more than one class.
            The function ignores it when ``n_classes`` is ``1``.
        bbox_format (BBoxFormatType): The format of the boxes in
            ``preds``.
        max_det (int): The maximum number of detections for each image.
        predicts_objectness (bool): Whether the confidence column holds a
            predicted objectness.

    Returns:
        ``list[Tensor]``: One tensor for each image, of shape
        ``[K, 6 + E]``, where ``K`` is the number of kept detections. A
        row holds the ``xyxy`` box, the score, the class index as a float,
        and the extra values. The rows go from the highest score to the
        lowest. An image without detections gets a tensor of shape
        ``[0, M]``.

    Raises:
        ValueError: When ``conf_thres`` or ``iou_thres`` is outside
            ``[0, 1]``.

    Example:
        The IoU of the first two boxes is ``0.81``, so NMS drops the box
        with the lower score:

        >>> import torch
        >>> preds = torch.tensor(
        ...     [
        ...         [
        ...             [0.0, 0.0, 10.0, 10.0, 0.75, 1.0],
        ...             [1.0, 1.0, 10.0, 10.0, 0.625, 1.0],
        ...             [20.0, 20.0, 30.0, 30.0, 0.5, 1.0],
        ...         ]
        ...     ]
        ... )
        >>> non_max_suppression(preds, n_classes=1)[0].tolist()
        [[0.0, 0.0, 10.0, 10.0, 0.75, 0.0],
         [20.0, 20.0, 30.0, 30.0, 0.5, 0.0]]

    """
    _validate_nms_thresholds(conf_thres, iou_thres)

    multi_label &= n_classes > 1

    # True when extra values follow the class scores.
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
        curr_out = _nms_single_image(
            x,
            candidate_mask[i],
            n_classes=n_classes,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            keep_classes=keep_classes,
            agnostic=agnostic,
            multi_label=multi_label,
            bbox_format=bbox_format,
            max_det=max_det,
            predicts_objectness=predicts_objectness,
            has_additional=has_additional,
        )
        if curr_out is not None:
            output[i] = curr_out

    return output


def anchors_for_fpn_features(
    features: list[Tensor],
    strides: Tensor,
    grid_cell_size: float = 5.0,
    grid_cell_offset: float = 0.5,
    multiply_with_stride: bool = False,
) -> tuple[Tensor, Tensor, list[int], Tensor]:
    """Generate the anchors of the feature maps of a feature pyramid.

    The function puts one anchor in each cell of each feature map. The
    anchor point of the cell in row ``i`` and column ``j`` is
    ``(j + grid_cell_offset, i + grid_cell_offset)``, in cells. The
    anchor box is a square with the side ``grid_cell_size * stride``
    around the anchor point. The side does not change with
    ``multiply_with_stride``. The anchors of the feature maps follow the
    order of ``features``, and go row by row in each map.

    The function pairs ``features`` with ``strides`` in order, and
    ignores the extra items of the longer one. The returned tensors have
    the dtype of the feature maps and are on the device of
    ``features[0]``.

    Args:
        features (``list[Tensor]``): The feature maps, each of shape
            ``[B, C, H, W]``. The function reads only their shapes and
            dtypes, and the device of the first map.
        strides (``Tensor``): One stride for each feature map, as a 1D
            tensor.
        grid_cell_size (float): The side of an anchor box, in strides.
        grid_cell_offset (float): The offset of an anchor point from the
            top-left corner of its cell, in cells.
        multiply_with_stride (bool): Whether to multiply the anchor points
            by the stride. The points are then in input image pixels.

    Returns:
        ``tuple[Tensor, Tensor, list[int], Tensor]``: Four values, where
        ``A`` is the total number of anchors:

        - the anchor boxes in ``xyxy`` format, of shape ``[A, 4]``;
        - the anchor points ``(x, y)``, of shape ``[A, 2]``;
        - the number of anchors ``H * W`` of each feature map;
        - the stride of each anchor, of shape ``[A, 1]``.

    Example:
        >>> import torch
        >>> features = [torch.zeros(1, 8, 2, 2), torch.zeros(1, 8, 1, 1)]
        >>> anchors, points, n_anchors, strides = anchors_for_fpn_features(
        ...     features, torch.tensor([8, 16]), multiply_with_stride=True
        ... )
        >>> n_anchors
        [4, 1]
        >>> points.tolist()
        [[4.0, 4.0], [12.0, 4.0], [4.0, 12.0], [12.0, 12.0], [8.0, 8.0]]
        >>> anchors[0].tolist()
        [-16.0, -16.0, 24.0, 24.0]
        >>> strides.flatten().tolist()
        [8.0, 8.0, 8.0, 8.0, 16.0]

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
    r"""Return the masks with zeros outside of their boxes.

    Mask ``i`` keeps the pixel in column :math:`x` and row :math:`y` when
    :math:`x_1 \le x < x_2` and :math:`y_1 \le y < y_2`, where
    :math:`(x_1, y_1, x_2, y_2)` is box ``i``. The comparison uses the
    integer indices of the pixels. The function does not change
    ``masks``.

    Args:
        masks (``Tensor``): The masks, of shape ``[N, H, W]``.
        bounding_boxes (``Tensor``): One ``xyxy`` box for each mask, in
            mask pixels, of shape ``[N, 4]``.

    Returns:
        ``Tensor``: The masks multiplied by the box regions, of shape
        ``[N, H, W]``.

    Example:
        >>> import torch
        >>> masks = torch.ones(1, 4, 4)
        >>> boxes = torch.tensor([[1.0, 0.0, 3.0, 2.0]])
        >>> apply_bounding_box_to_masks(masks, boxes)[0].int().tolist()
        [[0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

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
    r"""Compute the IoU loss between predicted boxes and target boxes.

    The function pairs each box of ``pred_bboxes`` with the box at the
    same position in ``target_bboxes``. It uses only the positive pairs
    when ``mask_positive`` is set. The loss of a pair is
    :math:`w (1 - \text{IoU})`. The weight :math:`w` is the sum of the
    target scores of the pair, or ``1`` without ``target_scores``.
    ``reduction`` selects the result:

    - ``"mean"``: the mean loss of the pairs.
    - ``"sum"``: the sum of the losses. When the sum of all values in
      ``target_scores`` is greater than ``1``, the function divides the
      result by that sum. This sum includes the pairs outside of
      ``mask_positive``.

    When ``mask_positive`` has no positive pair, the function returns a
    zero loss at once. It then checks neither ``reduction`` nor
    ``iou_type``.

    Args:
        pred_bboxes (``Tensor``): The predicted boxes, of shape
            ``[B, N, 4]``. Without ``mask_positive``, any shape
            ``[..., 4]`` works.
        target_bboxes (``Tensor``): The target boxes, of the same shape as
            ``pred_bboxes``. The function moves them to the device of
            ``pred_bboxes``.
        target_scores (``Tensor | None``): The target class scores, of
            shape ``[B, N, n_classes]``. ``None`` gives each pair the
            weight ``1``.
        mask_positive (``Tensor | None``): The boolean mask of the
            positive pairs, of shape ``[B, N]``. ``None`` uses all pairs.
        iou_type (IoUType): The IoU variant. See `bbox_iou`.
        bbox_format (BBoxFormatType): The format of both sets of boxes.
        reduction (``Literal["sum", "mean"]``): The reduction of the losses
            of the pairs.

    Returns:
        ``tuple[Tensor, Tensor]``: The scalar loss, and the detached IoU of
        each used pair, clamped to at least ``0``, of shape ``[K, 1]``.
        ``K`` is the number of used pairs. Without a positive pair, the
        IoU values are zeros of shape ``[B]``.

    Raises:
        ValueError: When ``reduction`` is not ``"sum"`` or ``"mean"``, or
            when ``iou_type`` is not a supported variant.
        NotImplementedError: When ``reduction`` is ``"sum"`` and
            ``target_scores`` is ``None``.

    Example:
        >>> import torch
        >>> pred = torch.tensor([[0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 2.0, 1.0]])
        >>> target = torch.tensor([[0.0, 0.0, 2.0, 2.0]]).repeat(2, 1)
        >>> loss, iou = compute_iou_loss(pred, target)
        >>> loss.item(), iou.tolist()
        (0.25, [[1.0], [0.5]])

        Without a positive pair, the loss is zero:

        >>> mask = torch.zeros(1, 2, dtype=torch.bool)
        >>> loss, iou = compute_iou_loss(
        ...     pred[None], target[None], mask_positive=mask
        ... )
        >>> loss.item(), iou.tolist()
        (0.0, [0.0])

    """
    device = pred_bboxes.device
    target_bboxes = target_bboxes.to(device)
    if mask_positive is not None and mask_positive.sum() == 0:
        return _empty_iou_loss(pred_bboxes, target_bboxes)
    loss_iou, iou = _positive_iou_loss(
        pred_bboxes,
        target_bboxes,
        target_scores,
        mask_positive,
        iou_type,
        bbox_format,
        reduction,
    )

    return loss_iou, iou.detach().clamp(0)


def keypoints_to_bboxes(
    keypoints: list[Tensor],
    img_height: int,
    img_width: int,
    box_width: int = 5,
    visibility_threshold: float = 0.5,
) -> list[Tensor]:
    """Convert keypoints to square boxes in ``xyxy`` format.

    The function drops the keypoints with a visibility below
    ``visibility_threshold``. It puts a square with the side
    ``box_width`` around each kept keypoint. It clips the top-left
    corner of the square at ``0``, and the bottom-right corner at
    ``img_width`` and ``img_height``.

    Args:
        keypoints (``list[Tensor]``): The keypoints of each image, each of
            shape ``[N, 1, 4]``. The values of a keypoint are
            ``(x, y, visibility, class_id)``.
        img_height (int): The image height, in pixels.
        img_width (int): The image width, in pixels.
        box_width (int): The side of a box, in pixels.
        visibility_threshold (float): The minimum visibility of a kept
            keypoint.

    Returns:
        ``list[Tensor]``: The boxes of each image, each of shape
        ``[K, 6]``, where ``K`` is the number of kept keypoints of the
        image. The values of a box are
        ``(x_min, y_min, x_max, y_max, visibility, class_id)``. An image
        without kept keypoints gets a tensor of shape ``[0, 6]``.

    Example:
        The function clips the box of the second keypoint at the image
        border. It drops the third keypoint, because its visibility is
        below ``0.5``:

        >>> import torch
        >>> keypoints = torch.tensor(
        ...     [
        ...         [[10.0, 10.0, 0.75, 2.0]],
        ...         [[1.0, 30.0, 0.5, 0.0]],
        ...         [[20.0, 20.0, 0.25, 1.0]],
        ...     ]
        ... )
        >>> boxes = keypoints_to_bboxes(
        ...     [keypoints], img_height=32, img_width=32
        ... )
        >>> boxes[0].tolist()
        [[7.5, 7.5, 12.5, 12.5, 0.75, 2.0],
         [0.0, 27.5, 3.5, 32.0, 0.5, 0.0]]

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


def _validate_nms_thresholds(conf_thres: float, iou_thres: float) -> None:
    if not (0 <= conf_thres <= 1):
        raise ValueError(
            f"Confidence threshold must be in range [0,1] but set to {conf_thres}."
        )
    if not (0 <= iou_thres <= 1):
        raise ValueError(
            f"IoU threshold must be in range [0,1] but set to {iou_thres}."
        )


def _apply_objectness(
    curr_out: Tensor, n_classes: int, predicts_objectness: bool
) -> Tensor:
    if predicts_objectness and n_classes == 1:
        curr_out[:, 5 : 5 + n_classes] = curr_out[:, 4:5]
    else:
        curr_out[:, 5 : 5 + n_classes] *= curr_out[:, 4:5]
    return curr_out


def _select_detections(
    curr_out: Tensor,
    bboxes: Tensor,
    n_classes: int,
    conf_thres: float,
    multi_label: bool,
) -> tuple[Tensor, Tensor]:
    if multi_label:
        # A box repeats once for each of its classes above the threshold.
        keep_idx, class_idx = (
            (curr_out[:, 5 : 5 + n_classes] > conf_thres)
            .nonzero(as_tuple=False)
            .T
        )
        curr_out = torch.cat(
            (
                bboxes[keep_idx],
                curr_out[keep_idx, class_idx + 5, None],
                class_idx[:, None].float(),
            ),
            1,
        )
    else:
        conf, class_idx = curr_out[:, 5 : 5 + n_classes].max(1, keepdim=True)
        keep_idx = (conf.view(-1) > conf_thres).nonzero().view(-1)
        curr_out = torch.cat((bboxes, conf, class_idx.float()), 1)[keep_idx]
    return curr_out, keep_idx


def _filter_keep_classes(curr_out: Tensor, keep_classes: list[int]) -> Tensor:
    return curr_out[
        (
            curr_out[:, 5:6]
            == torch.tensor(keep_classes, device=curr_out.device)
        ).any(1)
    ]


def _run_batched_nms(
    curr_out: Tensor, iou_thres: float, agnostic: bool, max_det: int
) -> Tensor:
    keep_indices = batched_nms(
        boxes=curr_out[:, :4],
        scores=curr_out[:, 4],
        iou_threshold=iou_thres,
        idxs=curr_out[:, 5].int() * (0 if agnostic else 1),
    )
    return curr_out[keep_indices[:max_det]]


def _nms_single_image(
    x: Tensor,
    candidate_mask_i: Tensor,
    *,
    n_classes: int,
    conf_thres: float,
    iou_thres: float,
    keep_classes: list[int] | None,
    agnostic: bool,
    multi_label: bool,
    bbox_format: BBoxFormatType,
    max_det: int,
    predicts_objectness: bool,
    has_additional: bool,
) -> Tensor | None:
    """Run the NMS steps of `non_max_suppression` on one image.

    Args:
        x (``Tensor``): The predictions of the image, of shape ``[N, M]``.
        candidate_mask_i (``Tensor``): The boolean mask of the rows of
            ``x`` that pass the confidence check of `non_max_suppression`,
            of shape ``[N]``.
        n_classes (int): The number of class score columns in ``x``.
        conf_thres (float): The score threshold.
        iou_thres (float): The IoU threshold of NMS.
        keep_classes (list[int] | None): The indices of the classes to
            keep. ``None`` keeps all classes.
        agnostic (bool): Whether NMS compares the boxes of different
            classes.
        multi_label (bool): Whether a box can get more than one class.
        bbox_format (BBoxFormatType): The format of the boxes in ``x``.
        max_det (int): The maximum number of detections.
        predicts_objectness (bool): Whether the confidence column holds a
            predicted objectness.
        has_additional (bool): Whether ``x`` has extra columns after the
            class scores.

    Returns:
        ``Tensor | None``: The kept detections in the format of the
        result of `non_max_suppression`, or ``None`` when no detection
        remains.

    """
    curr_out = x[candidate_mask_i]
    if curr_out.size(0) == 0:
        return None

    curr_out = _apply_objectness(curr_out, n_classes, predicts_objectness)

    bboxes = curr_out[:, :4]
    if bbox_format != "xyxy":
        bboxes = box_convert(bboxes, in_fmt=bbox_format, out_fmt="xyxy")

    curr_out, keep_idx = _select_detections(
        curr_out, bboxes, n_classes, conf_thres, multi_label
    )

    if has_additional:
        curr_out = torch.hstack(
            [curr_out, x[candidate_mask_i][keep_idx, 5 + n_classes :]]
        )

    if keep_classes is not None:
        curr_out = _filter_keep_classes(curr_out, keep_classes)

    if not curr_out.size(0):
        return None

    return _run_batched_nms(curr_out, iou_thres, agnostic, max_det)


def _empty_iou_loss(
    pred_bboxes: Tensor, target_bboxes: Tensor
) -> tuple[Tensor, Tensor]:
    return (
        torch.tensor(0.0).to(pred_bboxes.device),
        torch.zeros([target_bboxes.shape[0]]).to(pred_bboxes.device),
    )


def _positive_iou_loss(
    pred_bboxes: Tensor,
    target_bboxes: Tensor,
    target_scores: Tensor | None,
    mask_positive: Tensor | None,
    iou_type: IoUType,
    bbox_format: BBoxFormatType,
    reduction: Literal["sum", "mean"],
) -> tuple[Tensor, Tensor]:
    bbox_mask = _bbox_mask(pred_bboxes, mask_positive)
    bbox_weight = _bbox_weight(target_scores, mask_positive)
    iou = bbox_iou(
        torch.masked_select(pred_bboxes, bbox_mask).reshape([-1, 4]),
        torch.masked_select(target_bboxes, bbox_mask).reshape([-1, 4]),
        iou_type=iou_type,
        bbox_format=bbox_format,
        element_wise=True,
    ).unsqueeze(-1)
    return _reduce_iou_loss(
        (1 - iou) * bbox_weight, target_scores, reduction
    ), iou


def _bbox_mask(pred_bboxes: Tensor, mask_positive: Tensor | None) -> Tensor:
    if mask_positive is None:
        return torch.ones_like(pred_bboxes, dtype=torch.bool)
    return mask_positive.unsqueeze(-1).repeat([1, 1, 4])


def _bbox_weight(
    target_scores: Tensor | None, mask_positive: Tensor | None
) -> Tensor:
    if target_scores is None:
        return torch.tensor(1.0)
    mask = (
        mask_positive
        if mask_positive is not None
        else torch.ones_like(target_scores.sum(-1), dtype=torch.bool)
    )
    return torch.masked_select(target_scores.sum(-1), mask).unsqueeze(-1)


def _reduce_iou_loss(
    loss_iou: Tensor,
    target_scores: Tensor | None,
    reduction: Literal["sum", "mean"],
) -> Tensor:
    if reduction == "mean":
        return loss_iou.mean()
    if reduction != "sum":
        raise ValueError(f"Unknown reduction type `{reduction}`")
    if target_scores is None:
        raise NotImplementedError(
            "Sum reduction is not supported when `target_scores` is None"
        )
    loss_iou = loss_iou.sum()
    return (
        loss_iou / target_scores.sum() if target_scores.sum() > 1 else loss_iou
    )
