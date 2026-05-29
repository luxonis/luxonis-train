import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .utils import batch_iou, bbox_iou, candidates_in_gt, fix_collisions


class ATSSAssigner(nn.Module):
    def __init__(self, n_classes: int, topk: int = 9):
        """Initialize the Adaptive Training Sample Selection assigner.

        Adapted from `Bridging the Gap Between Anchor-based and Anchor-free
        Detection via Adaptive Training Sample Selection
        <https://arxiv.org/pdf/1912.02424.pdf>`_. Code is adapted from
        `PPYOLOE_pytorch <https://github.com/Nioolek/PPYOLOE_pytorch/blob/master/ppyoloe/assigner/atss_assigner.py>`_
        and `TOOD <https://github.com/fcjian/TOOD/blob/master/mmdet/core/bbox/assigners/atss_assigner.py>`_.

        Args:
            n_classes (int): Number of classes in the dataset.
            topk (int): Number of anchors considered in selection.
                Defaults to ``9``.
        """
        super().__init__()

        self.topk = topk
        self.n_classes = n_classes

    def forward(
        self,
        anchor_bboxes: Tensor,
        n_level_bboxes: list[int],
        gt_labels: Tensor,
        gt_bboxes: Tensor,
        mask_gt: Tensor,
        pred_bboxes: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Generate final assignments.

        Args:
            anchor_bboxes (Tensor): Anchor bboxes with shape
                ``[n_anchors, 4]``.
            n_level_bboxes (list[int]): Number of bboxes per level.
            gt_labels (Tensor): Initial GT labels with shape
                ``[bs, n_max_boxes, 1]``.
            gt_bboxes (Tensor): Initial GT bboxes with shape
                ``[bs, n_max_boxes, 4]``.
            mask_gt (Tensor): Mask for valid GTs with shape
                ``[bs, n_max_boxes, 1]``.
            pred_bboxes (Tensor): Predicted bboxes with shape
                ``[bs, n_anchors, 4]``.

        Returns:
            tuple[Tensor, Tensor, Tensor, Tensor, Tensor]: Assigned labels
            with shape ``[bs, n_anchors]``, assigned bboxes with shape
            ``[bs, n_anchors, 4]``, assigned scores with shape
            ``[bs, n_anchors, n_classes]``, output positive mask with shape
            ``[bs, n_anchors]``, and assigned GT indices with shape
            ``[bs, n_anchors]``.
        """
        self.n_anchors = anchor_bboxes.size(0)
        self.bs = gt_bboxes.size(0)
        self.n_max_boxes = gt_bboxes.size(1)

        if self.n_max_boxes == 0:
            device = gt_bboxes.device
            return (
                torch.full([self.bs, self.n_anchors], self.n_classes).to(
                    device
                ),
                torch.zeros([self.bs, self.n_anchors, 4]).to(device),
                torch.zeros([self.bs, self.n_anchors, self.n_classes]).to(
                    device
                ),
                torch.zeros([self.bs, self.n_anchors]).to(device),
                torch.zeros([self.bs, self.n_anchors]).to(device),
            )

        gt_bboxes_flat = gt_bboxes.reshape([-1, 4])

        # Compute iou between all gt and anchor bboxes
        overlaps = bbox_iou(gt_bboxes_flat, anchor_bboxes)
        overlaps = overlaps.reshape([self.bs, -1, self.n_anchors])

        # Compute center distance between all gt and anchor bboxes
        gt_centers = self._get_bbox_center(gt_bboxes_flat)
        anchor_centers = self._get_bbox_center(anchor_bboxes)
        distances = (
            (gt_centers[:, None, :] - anchor_centers[None, :, :])
            .pow(2)
            .sum(-1)
            .sqrt()
        )
        distances = distances.reshape([self.bs, -1, self.n_anchors])

        # Select candidates based on the center distance
        is_in_topk, topk_idxs = self._select_topk_candidates(
            distances, n_level_bboxes, mask_gt
        )

        # Compute threshold and selected positive candidates based on it
        is_pos = self._get_positive_samples(is_in_topk, topk_idxs, overlaps)

        # Select candidates inside GT
        is_in_gts = candidates_in_gt(anchor_centers, gt_bboxes_flat)
        is_in_gts = is_in_gts.reshape(self.bs, self.n_max_boxes, -1)

        # Final positive candidates
        mask_pos = is_pos * is_in_gts * mask_gt

        # If an anchor box is assigned to multiple gts, the one with the highest IoU is selected
        assigned_gt_idx, mask_pos_sum, mask_pos = fix_collisions(
            mask_pos, overlaps, self.n_max_boxes
        )

        # Generate final assignments based on masks
        (assigned_labels, assigned_bboxes, assigned_scores) = (
            self._get_final_assignments(
                gt_labels, gt_bboxes, assigned_gt_idx, mask_pos_sum
            )
        )

        # Soft label with IoU
        ious = batch_iou(gt_bboxes, pred_bboxes) * mask_pos
        ious = ious.max(dim=-2)[0].unsqueeze(-1)
        assigned_scores *= ious

        out_mask_positive = mask_pos_sum.bool()

        return (
            assigned_labels.long(),
            assigned_bboxes,
            assigned_scores,
            out_mask_positive,
            assigned_gt_idx,
        )

    def _get_bbox_center(self, bbox: Tensor) -> Tensor:
        """Compute centers of bboxes.

        Args:
            bbox (Tensor): Bounding boxes with shape ``[N, 4]``.

        Returns:
            Tensor: Bounding-box centers with shape ``[N, 2]``.
        """
        cx = (bbox[:, 0] + bbox[:, 2]) / 2.0
        cy = (bbox[:, 1] + bbox[:, 3]) / 2.0
        return torch.stack((cx, cy), dim=1).to(bbox.device)

    def _select_topk_candidates(
        self, distances: Tensor, n_level_bboxes: list[int], mask_gt: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Select k anchors whose centers are closest to GT.

        Args:
            distances (Tensor): Distances between GT and anchor centers with
                shape ``[bs, n_max_boxes, n_anchors]``.
            n_level_bboxes (list[int]): Number of bboxes per level.
            mask_gt (Tensor): Mask for valid GTs per image with shape
                ``[bs, n_max_boxes, 1]``.

        Returns:
            tuple[Tensor, Tensor]: A tuple ``(is_in_topk, topk_idxs)``
            containing the selected-anchor mask and selected-anchor indices.
        """
        mask_gt = mask_gt.repeat(1, 1, self.topk).bool()
        level_distances = distances.split(n_level_bboxes, dim=-1)
        is_in_topk_list: list[Tensor] = []
        topk_idxs: list[Tensor] = []
        start_idx = 0
        for per_level_distances, per_level_boxes in zip(
            level_distances, n_level_bboxes, strict=True
        ):
            end_idx = start_idx + per_level_boxes
            selected_k = min(self.topk, per_level_boxes)
            _, per_level_topk_idxs = per_level_distances.topk(
                selected_k, dim=-1, largest=False
            )
            topk_idxs.append(per_level_topk_idxs + start_idx)
            per_level_topk_idxs = torch.where(
                mask_gt,
                per_level_topk_idxs,
                torch.zeros_like(per_level_topk_idxs),
            )
            is_in_topk = F.one_hot(per_level_topk_idxs, per_level_boxes).sum(
                dim=-2
            )
            is_in_topk = torch.where(
                is_in_topk > 1, torch.zeros_like(is_in_topk), is_in_topk
            )
            is_in_topk_list.append(is_in_topk.to(distances.dtype))
            start_idx = end_idx

        return torch.cat(is_in_topk_list, dim=-1), torch.cat(topk_idxs, dim=-1)

    def _get_positive_samples(
        self, is_in_topk: Tensor, topk_idxs: Tensor, overlaps: Tensor
    ) -> Tensor:
        """Compute threshold and return mask for samples over threshold.

        Args:
            is_in_topk (Tensor): Mask of selected anchors with shape
                ``[bs, n_max_boxes, n_anchors]``.
            topk_idxs (Tensor): Indices of selected anchors with shape
                ``[bs, n_max_boxes, topk * n_levels]``.
            overlaps (Tensor): IoUs between GTs and anchors with shape
                ``[bs, n_max_boxes, n_anchors]``.

        Returns:
            Tensor: Mask of positive samples with shape
            ``[bs, n_max_boxes, n_anchors]``.
        """
        n_bs_max_boxes = self.bs * self.n_max_boxes
        _candidate_overlaps = torch.where(
            is_in_topk > 0, overlaps, torch.zeros_like(overlaps)
        )
        topk_idxs = topk_idxs.reshape([n_bs_max_boxes, -1])
        assist_idxs = self.n_anchors * torch.arange(
            n_bs_max_boxes, device=topk_idxs.device
        )
        assist_idxs = assist_idxs[:, None]
        flatten_idxs = topk_idxs + assist_idxs
        candidate_overlaps = _candidate_overlaps.reshape(-1)[flatten_idxs]
        candidate_overlaps = candidate_overlaps.reshape(
            [self.bs, self.n_max_boxes, -1]
        )

        overlaps_mean_per_gt = candidate_overlaps.mean(dim=-1, keepdim=True)
        overlaps_std_per_gt = candidate_overlaps.std(dim=-1, keepdim=True)
        overlaps_thr_per_gt = overlaps_mean_per_gt + overlaps_std_per_gt

        return torch.where(
            _candidate_overlaps
            > overlaps_thr_per_gt.repeat([1, 1, self.n_anchors]),
            is_in_topk,
            torch.zeros_like(is_in_topk),
        )

    def _get_final_assignments(
        self,
        gt_labels: Tensor,
        gt_bboxes: Tensor,
        assigned_gt_idx: Tensor,
        mask_pos_sum: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Generate final assignments based on the mask.

        Args:
            gt_labels (Tensor): Initial GT labels with shape
                ``[bs, n_max_boxes, 1]``.
            gt_bboxes (Tensor): Initial GT bboxes with shape
                ``[bs, n_max_boxes, 4]``.
            assigned_gt_idx (Tensor): Indices of matched GTs with shape
                ``[bs, n_anchors]``.
            mask_pos_sum (Tensor): Mask of matched GTs with shape
                ``[bs, n_anchors]``.

        Returns:
            tuple[Tensor, Tensor, Tensor]: Assigned labels with shape
            ``[bs, n_anchors]``, assigned bboxes with shape
            ``[bs, n_anchors, 4]``, and assigned scores with shape
            ``[bs, n_anchors, n_classes]``.
        """
        # assigned target labels
        batch_idx = torch.arange(
            self.bs, dtype=gt_labels.dtype, device=gt_labels.device
        )
        batch_idx = batch_idx[..., None]
        assigned_gt_idx = (
            assigned_gt_idx + batch_idx * self.n_max_boxes
        ).long()
        assigned_labels = gt_labels.flatten()[assigned_gt_idx.flatten()]
        assigned_labels = assigned_labels.reshape([self.bs, self.n_anchors])
        assigned_labels = torch.where(
            mask_pos_sum > 0,
            assigned_labels,
            torch.full_like(assigned_labels, self.n_classes),
        )

        # assigned target boxes
        assigned_bboxes = gt_bboxes.reshape([-1, 4])[assigned_gt_idx.flatten()]
        assigned_bboxes = assigned_bboxes.reshape([self.bs, self.n_anchors, 4])

        # assigned target scores
        assigned_scores = F.one_hot(
            assigned_labels.long(), self.n_classes + 1
        ).float()
        assigned_scores = assigned_scores[:, :, : self.n_classes]

        return assigned_labels, assigned_bboxes, assigned_scores
