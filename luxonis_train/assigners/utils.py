import torch
import torch.nn.functional as F
from torch import Tensor

from luxonis_train.utils import bbox_iou


def candidates_in_gt(
    anchor_centers: Tensor, gt_bboxes: Tensor, eps: float = 1e-9
) -> Tensor:
    """Check if anchor box's center is in any GT bbox.

    @type anchor_centers: Tensor
    @param anchor_centers: Centers of anchor bboxes [n_anchors, 2]
    @type gt_bboxes: Tensor
    @param gt_bboxes: Ground truth bboxes [bs * n_max_boxes, 4]
    @type eps: float
    @param eps: Threshold for minimum delta. Defaults to 1e-9.
    @rtype: Tensor
    @return: Mask for anchors inside any GT bbox
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
    candidates = (bbox_delta.min(dim=-1)[0] > eps).to(gt_bboxes.dtype)
    return candidates


def fix_collisions(
    mask_pos: Tensor, overlaps: Tensor, n_max_boxes: int
) -> tuple[Tensor, Tensor, Tensor]:
    """If an anchor is assigned to multiple GTs, the one with highest
    IoU is selected.

    @type mask_pos: Tensor
    @param mask_pos: Mask of assigned anchors [bs, n_max_boxes,
        n_anchors]
    @type overlaps: Tensor
    @param overlaps: IoUs between GTs and anchors [bx, n_max_boxes,
        n_anchors]
    @type n_max_boxes: int
    @param n_max_boxes: Number of maximum boxes per image
    @rtype: tuple[Tensor, Tensor, Tensor]
    @return: Assigned indices, sum of positive mask, positive mask
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
    """Calculates IoU for each pair of bounding boxes in the batch.
    Bounding boxes must be in the "xyxy" format.

    @type batch1: Tensor
    @param batch1: Tensor of shape C{[bs, N, 4]}
    @type batch2: Tensor
    @param batch2: Tensor of shape C{[bs, M, 4]}
    @rtype: Tensor
    @return: Per image box IoU of shape C{[bs, N, M]}
    """
    ious = torch.stack(
        [bbox_iou(batch1[i], batch2[i]) for i in range(batch1.size(0))], dim=0
    )
    return ious


def batch_pose_oks(
    gt_kps: torch.Tensor,
    pred_kps: torch.Tensor,
    gt_bboxes: torch.Tensor,
    kp_sigmas: torch.Tensor,
    eps: float = 1e-9,
    area_factor: float = 0.53,
) -> torch.Tensor:
    """Compute batched Object Keypoint Similarity (OKS) between ground
    truth and predicted keypoints.

    @type gt_kps: torch.Tensor
    @param gt_kps: Ground truth keypoints with shape [N, M1,
        num_keypoints, 3]
    @type pred_kps: torch.Tensor
    @param pred_kps: Predicted keypoints with shape [N, M1,
        num_keypoints, 3]
    @type gt_bboxes: torch.Tensor
    @param gt_bboxes: Ground truth bounding boxes in XYXY format with
        shape [N, M1, 4]
    @type kp_sigmas: torch.Tensor
    @param kp_sigmas: Sigmas for each keypoint, shape [num_keypoints]
    @type eps: float
    @param eps: A small constant to ensure numerical stability
    @rtype: torch.Tensor
    @return: A tensor of OKS values with shape [N, M1, M1]
    """

    gt_xy = gt_kps[:, :, :, :2].unsqueeze(
        2
    )  # shape: [N, M1, 1, num_keypoints, 2]
    pred_xy = pred_kps[:, :, :, :2].unsqueeze(
        1
    )  # shape: [N, 1, M1, num_keypoints, 2]

    sq_diff = ((gt_xy - pred_xy) ** 2).sum(
        dim=-1
    )  # shape: [N, M1, M1, num_keypoints]

    width = gt_bboxes[:, :, 2] - gt_bboxes[:, :, 0]
    height = gt_bboxes[:, :, 3] - gt_bboxes[:, :, 1]
    pose_area = (
        (width * height * area_factor).unsqueeze(-1).unsqueeze(-1)
    )  # shape: [N, M1, 1, 1]

    kp_sigmas = kp_sigmas.view(1, 1, 1, -1)  # shape: [1, 1, 1, num_keypoints]

    exp_term = sq_diff / ((2 * kp_sigmas) ** 2) / (pose_area + eps) / 2
    oks_vals = torch.exp(-exp_term)  # shape: [N, M1, M1, num_keypoints]

    vis_mask = (
        gt_kps[:, :, :, 2].gt(0).float().unsqueeze(2)
    )  # shape: [N, M1, 1, num_keypoints]
    vis_count = vis_mask.sum(dim=-1)  # shape: [N, M1, M1]

    mean_oks = (oks_vals * vis_mask).sum(dim=-1) / (
        vis_count + eps
    )  # shape: [N, M1, M1]

    return mean_oks
