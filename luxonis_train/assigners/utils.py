"""Helpers that `ATSSAssigner` and `TaskAlignedAssigner` share.

The helpers find the anchors inside each box, resolve the anchors that
are positive for more than one box, and compute the IoU for a batch of
images.

"""

import torch
import torch.nn.functional as F
from torch import Tensor

from luxonis_train.utils import bbox_iou


def candidates_in_gt(
    anchor_centers: Tensor, gt_bboxes: Tensor, eps: float = 1e-9
) -> Tensor:
    """Mark the anchors that have a center inside each box.

    A center is inside a box when its distance to each of the four sides
    of the box is larger than ``eps``. A center on a side is outside.

    Args:
        anchor_centers (``Tensor``): Anchor centers ``(x, y)`` with shape
            ``[n_anchors, 2]``.
        gt_bboxes (``Tensor``): Boxes in ``xyxy`` format with shape
            ``[n_boxes, 4]``. The assigners give the ground truth boxes
            of all images as one flat tensor, so ``n_boxes`` is
            ``bs * n_max_boxes``.
        eps (float): The value that the distance between a center and
            each side of the box must exceed.

    Returns:
        ``Tensor``: Mask with shape ``[n_boxes, n_anchors]`` and the dtype
        of ``gt_bboxes``. ``1`` marks a center inside the box.

    Example:
        >>> import torch
        >>> centers = torch.tensor([[1.0, 1.0], [3.0, 3.0], [2.0, 0.0]])
        >>> boxes = torch.tensor([[0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 4.0, 4.0]])
        >>> candidates_in_gt(centers, boxes).tolist()
        [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]

    """
    n_anchors = anchor_centers.size(0)
    anchor_centers = anchor_centers.unsqueeze(0).repeat(
        gt_bboxes.size(0), 1, 1
    )
    gt_bboxes_lt = gt_bboxes[:, :2].unsqueeze(1).repeat(1, n_anchors, 1)
    gt_bboxes_rb = gt_bboxes[:, 2:].unsqueeze(1).repeat(1, n_anchors, 1)
    bbox_delta_lt = anchor_centers - gt_bboxes_lt
    bbox_delta_rb = gt_bboxes_rb - anchor_centers
    bbox_delta = torch.cat([bbox_delta_lt, bbox_delta_rb], dim=-1)
    return (bbox_delta.min(dim=-1)[0] > eps).to(gt_bboxes.dtype)


def fix_collisions(
    mask_pos: Tensor, overlaps: Tensor, n_max_boxes: int
) -> tuple[Tensor, Tensor, Tensor]:
    """Resolve the anchors that are positive for more than one box.

    Such an anchor goes to the box with the highest overlap. The search
    covers all ``n_max_boxes`` boxes of the image, not only the boxes
    for which the anchor is positive. The mask of the other anchors does
    not change.

    Args:
        mask_pos (``Tensor``): Positive mask with shape
            ``[bs, n_max_boxes, n_anchors]``. ``1`` marks an anchor that
            is positive for a box.
        overlaps (``Tensor``): The overlap of each box and anchor with
            shape ``[bs, n_max_boxes, n_anchors]``. `ATSSAssigner` gives
            the IoU with the anchor boxes. `TaskAlignedAssigner` gives the
            IoU with the predicted boxes, times the object keypoint
            similarity when it gets keypoints.
        n_max_boxes (int): Number of box slots per image, the size of
            dimension ``1`` of ``mask_pos``.

    Returns:
        ``tuple[Tensor, Tensor, Tensor]``: Three tensors:

        - ``assigned_gt_idx`` (``[bs, n_anchors]``, ``int64``): The index
          of the box of each anchor, ``0`` for an anchor without a box.
        - ``mask_pos_sum`` (``[bs, n_anchors]``): The number of boxes of
          each anchor, ``0`` or ``1``.
        - ``mask_pos`` (``[bs, n_max_boxes, n_anchors]``): The positive
          mask with at most one box for each anchor.

    Example:
        The second anchor is positive for both boxes. It has the higher
        overlap with the second box.

        >>> import torch
        >>> mask_pos = torch.tensor([[[1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]])
        >>> overlaps = torch.tensor([[[0.9, 0.3, 0.1], [0.2, 0.6, 0.0]]])
        >>> gt_idx, mask_pos_sum, mask_pos = fix_collisions(
        ...     mask_pos, overlaps, 2
        ... )
        >>> gt_idx.tolist()
        [[0, 1, 0]]
        >>> mask_pos_sum.tolist()
        [[1.0, 1.0, 0.0]]
        >>> mask_pos.tolist()
        [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]

    """
    mask_pos_sum = mask_pos.sum(dim=-2)
    if mask_pos_sum.max() > 1:
        mask_multi_gts = (mask_pos_sum.unsqueeze(1) > 1).repeat(
            [1, n_max_boxes, 1]
        )
        max_overlaps_idx = overlaps.argmax(dim=1)
        is_max_overlaps = F.one_hot(max_overlaps_idx, n_max_boxes)
        is_max_overlaps = is_max_overlaps.permute(0, 2, 1).to(overlaps.dtype)
        mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos)
        mask_pos_sum = mask_pos.sum(dim=-2)
    assigned_gt_idx = mask_pos.argmax(dim=-2)
    return assigned_gt_idx, mask_pos_sum, mask_pos


def batch_iou(batch1: Tensor, batch2: Tensor) -> Tensor:
    """Compute the IoU between each pair of boxes in each image.

    The function calls `bbox_iou` once for each image of the batch.

    Args:
        batch1 (``Tensor``): Boxes in ``xyxy`` format with shape
            ``[bs, N, 4]``.
        batch2 (``Tensor``): Boxes in ``xyxy`` format with shape
            ``[bs, M, 4]``.

    Returns:
        ``Tensor``: IoU values with shape ``[bs, N, M]``. The value at
        ``[b, i, j]`` is the IoU between ``batch1[b, i]`` and
        ``batch2[b, j]``.

    Example:
        >>> import torch
        >>> batch1 = torch.tensor([[[0.0, 0.0, 2.0, 2.0]]])
        >>> batch2 = torch.tensor(
        ...     [[[0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 4.0, 2.0]]]
        ... )
        >>> batch_iou(batch1, batch2).tolist()
        [[[1.0, 0.5]]]

    """
    return torch.stack(
        [bbox_iou(batch1[i], batch2[i]) for i in range(batch1.size(0))], dim=0
    )
