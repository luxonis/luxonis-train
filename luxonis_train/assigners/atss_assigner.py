import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .utils import batch_iou, bbox_iou, candidates_in_gt, fix_collisions


class ATSSAssigner(nn.Module):
    def __init__(self, n_classes: int, topk: int = 9):
        """Adaptive Training Sample Selection Assigner, adapted
        from U{Bridging the Gap Between Anchor-based and Anchor-free Detection via
        Adaptive Training Sample Selection<https://arxiv.org/pdf/1912.02424.pdf>}.
        Code is adapted from: U{https://github.com/Nioolek/PPYOLOE_pytorch/blob/master/
        ppyoloe/assigner/atss_assigner.py} and
        U{https://github.com/fcjian/TOOD/blob/master/mmdet/core/bbox/
        assigners/atss_assigner.py}

        @type n_classes: int
        @param n_classes: Number of classes in the dataset.
        @type topk: int
        @param topk: Number of anchors considere in selection. Defaults to 9.
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
        """Assigner's forward method which generates final assignments.

        @type anchor_bboxes: Tensor
        @param anchor_bboxes: Anchor bboxes of shape [n_anchors, 4]
        @type n_level_bboxes: list[int]
        @param n_level_bboxes: Number of bboxes per level
        @type gt_labels: Tensor
        @param gt_labels: Initial GT labels [bs, n_max_boxes, 1]
        @type gt_bboxes: Tensor
        @param gt_bboxes: Initial GT bboxes [bs, n_max_boxes, 4]
        @type mask_gt: Tensor
        @param mask_gt: Mask for valid GTs [bs, n_max_boxes, 1]
        @type pred_bboxes: Tensor
        @param pred_bboxes: Predicted bboxes of shape [bs, n_anchors, 4]
        @rtype: tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
        @return: Assigned labels of shape [bs, n_anchors], assigned
            bboxes of shape [bs, n_anchors, 4], assigned scores of shape
            [bs, n_anchors, n_classes] and output positive mask of shape
            [bs, n_anchors].
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
        """Computes centers of bbox with shape [N,4]"""
        cx = (bbox[:, 0] + bbox[:, 2]) / 2.0
        cy = (bbox[:, 1] + bbox[:, 3]) / 2.0
        return torch.stack((cx, cy), dim=1).to(bbox.device)

    def _select_topk_candidates(
        self, distances: Tensor, n_level_bboxes: list[int], mask_gt: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Select k anchors whose centers are closest to GT.

        @type distance: Tensor
        @param distance: Distances between GT and anchor centers.
        @type n_level_bboxes: list[int]
        @param n_level_bboxes: list of number of bboxes per level.
        @type mask_gt: Tensor
        @param mask_gt: Mask for valid GT per image.
        @rtype: tuple[Tensor, Tensor]
        @return: Mask of selected anchors and indices of selected
            anchors.
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
        """Computes threshold and returns mask for samples over
        threshold.

        @type is_in_topk: Tensor
        @param is_in_topk: Mask of selected anchors [bx, n_max_boxes,
            n_anchors]
        @type topk_idxs: Tensor
        @param topk_idxs: Indices of selected anchors [bx, n_max_boxes,
            topK * n_levels]
        @type overlaps: Tensor
        @param overlaps: IoUs between GTs and anchors [bx, n_max_boxes,
            n_anchors]
        @rtype: Tensor
        @return: Mask of positive samples [bx, n_max_boxes, n_anchors]
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

        @type gt_labels: Tensor
        @param gt_labels: Initial GT labels [bs, n_max_boxes, 1]
        @type gt_bboxes: Tensor
        @param gt_bboxes: Initial GT bboxes [bs, n_max_boxes, 4]
        @type assigned_gt_idx: Tensor
        @param assigned_gt_idx: Indices of matched GTs [bs, n_anchors]
        @type mask_pos_sum: Tensor
        @param mask_pos_sum: Mask of matched GTs [bs, n_anchors]
        @rtype: tuple[Tensor, Tensor, Tensor]
        @return: Assigned labels of shape [bs, n_anchors], assigned
            bboxes of shape [bs, n_anchors, 4], assigned scores of shape
            [bs, n_anchors, n_classes].
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
