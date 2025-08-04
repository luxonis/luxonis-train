import torch
import torch.nn.functional as F
from luxonis_ml.typing import all_not_none, any_not_none
from torch import Tensor, nn

from luxonis_train.utils import compute_pose_oks

from .utils import batch_iou, candidates_in_gt, fix_collisions


class TaskAlignedAssigner(nn.Module):
    def __init__(
        self,
        n_classes: int,
        topk: int = 13,
        alpha: float = 1.0,
        beta: float = 6.0,
        eps: float = 1e-9,
    ):
        """Task Aligned Assigner.

        Adapted from: U{TOOD: Task-aligned One-stage Object Detection<https://arxiv.org/pdf/2108.07755.pdf>}.
        Code is adapted from: U{https://github.com/Nioolek/PPYOLOE_pytorch/blob/master/ppyoloe/assigner/tal_assigner.py}.

        @license: U{Apache License, Version 2.0<https://github.com/Nioolek/PPYOLOE_pytorch/
            tree/master?tab=Apache-2.0-1-ov-file#readme>}

        @type n_classes: int
        @param n_classes: Number of classes in the dataset.
        @type topk: int
        @param topk: Number of anchors considered in selection. Defaults to 13.
        @type alpha: float
        @param alpha: Defaults to 1.0.
        @type beta: float
        @param beta: Defaults to 6.0.
        @type eps: float
        @param eps: Defaults to 1e-9.
        """
        super().__init__()

        self.n_classes = n_classes
        self.topk = topk
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    @torch.no_grad()
    def forward(
        self,
        pred_scores: Tensor,
        pred_bboxes: Tensor,
        anchor_points: Tensor,
        gt_labels: Tensor,
        gt_bboxes: Tensor,
        mask_gt: Tensor,
        pred_kpts: Tensor | None = None,
        gt_kpts: Tensor | None = None,
        sigmas: Tensor | None = None,
        area_factor: float | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Assigner's forward method which generates final assignments.

        If both pred_kpts and gt_kpts are provided, a pose OKS is
        computed and used in the alignment metric; the final tuple then
        includes assigned poses.

        @type pred_scores: Tensor
        @param pred_scores: Predicted scores [bs, n_anchors, 1]
        @type pred_bboxes: Tensor
        @param pred_bboxes: Predicted bboxes [bs, n_anchors, 4]
        @type anchor_points: Tensor
        @param anchor_points: Anchor points [n_anchors, 2]
        @type gt_labels: Tensor
        @param gt_labels: Initial GT labels [bs, n_max_boxes, 1]
        @type gt_bboxes: Tensor
        @param gt_bboxes: Initial GT bboxes [bs, n_max_boxes, 4]
        @type mask_gt: Tensor
        @param mask_gt: Mask for valid GTs [bs, n_max_boxes, 1]
        @type pred_kpts: Tensor | None
        @param pred_kpts: Predicted keypoints [bs, n_anchors, n_kpts,
            3] (optional)
        @type gt_kpts: Tensor | None
        @param gt_kpts: Ground truth keypoints [bs, n_max_boxes,
            n_kpts, 3] (optional)
        @type sigmas: Tensor | None
        @param sigmas: Sigmas for OKS computation if keypoints are used.
        @type area_factor: float | None
        @param area_factor: Area factor for OKS computation. Defaults to
            0.53.
        @rtype: tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
        @return: Assigned labels of shape [bs, n_anchors], assigned
            bboxes of shape [bs, n_anchors, 4], assigned scores of shape
            [bs, n_anchors, n_classes] and output mask of shape [bs,
            n_anchors]
        """
        if any_not_none(
            [pred_kpts, gt_kpts, sigmas, area_factor]
        ) and not all_not_none([pred_kpts, gt_kpts, sigmas, area_factor]):
            raise ValueError(
                "All `pred_kpts`, `gt_kpts`, `sigmas`, and `area_factor` "
                "must be provided if OKS is to be computed, "
                "but only some of them have been provided."
            )

        self.bs = pred_scores.size(0)
        self.n_max_boxes = gt_bboxes.size(1)

        if self.n_max_boxes == 0:
            device = gt_bboxes.device
            return (
                torch.full_like(
                    pred_scores[..., 0], self.n_classes, dtype=torch.int64
                ).to(device),
                torch.zeros_like(pred_bboxes, dtype=gt_bboxes.dtype).to(
                    device
                ),
                torch.zeros_like(pred_scores, dtype=pred_scores.dtype).to(
                    device
                ),
                torch.zeros_like(pred_scores[..., 0], dtype=torch.bool).to(
                    device
                ),
                torch.zeros_like(pred_scores[..., 0], dtype=torch.int64).to(
                    device
                ),
            )

        # Compute alignment metric between all bboxes and optionally incorporate pose OKS
        align_metric, overlaps = self._get_alignment_metric(
            pred_scores,
            pred_bboxes,
            gt_labels,
            gt_bboxes,
            pred_kpts,
            gt_kpts,
            sigmas,
            area_factor,
        )

        # Select top-k bboxes as candidates for each GT
        is_in_gts = candidates_in_gt(anchor_points, gt_bboxes.reshape(-1, 4))
        is_in_gts = is_in_gts.reshape(self.bs, self.n_max_boxes, -1)
        is_in_topk = self._select_topk_candidates(
            align_metric * is_in_gts,
            topk_mask=mask_gt.repeat(1, 1, self.topk).bool(),
        )

        # Final positive candidates
        mask_pos = is_in_topk * is_in_gts * mask_gt

        # If an anchor box is assigned to multiple gts, the one with the highest IoU is selected
        assigned_gt_idx, mask_pos_sum, mask_pos = fix_collisions(
            mask_pos, overlaps, self.n_max_boxes
        )

        # Generate final targets based on masks
        (assigned_labels, assigned_bboxes, assigned_scores) = (
            self._get_final_assignments(
                gt_labels, gt_bboxes, assigned_gt_idx, mask_pos_sum
            )
        )

        # normalize
        align_metric *= mask_pos
        pos_align_metrics = align_metric.max(dim=-1, keepdim=True)[0]
        pos_overlaps = (overlaps * mask_pos).max(dim=-1, keepdim=True)[0]
        norm_align_metric = (
            (align_metric * pos_overlaps / (pos_align_metrics + self.eps))
            .max(-2)[0]
            .unsqueeze(-1)
        )
        assigned_scores = assigned_scores * norm_align_metric

        out_mask_positive = mask_pos_sum.bool()

        return (
            assigned_labels,
            assigned_bboxes,
            assigned_scores,
            out_mask_positive,
            assigned_gt_idx,
        )

    def _get_alignment_metric(
        self,
        pred_scores: Tensor,
        pred_bboxes: Tensor,
        gt_labels: Tensor,
        gt_bboxes: Tensor,
        pred_kpts: Tensor | None = None,
        gt_kpts: Tensor | None = None,
        sigmas: Tensor | None = None,
        area_factor: float | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Calculates anchor alignment metric and IoU between GTs and
        predicted bboxes (optionally incorporating pose OKS).

        @type pred_scores: Tensor
        @param pred_scores: Predicted scores [bs, n_anchors, 1]
        @type pred_bboxes: Tensor
        @param pred_bboxes: Predicted bboxes [bs, n_anchors, 4]
        @type gt_labels: Tensor
        @param gt_labels: Initial GT labels [bs, n_max_boxes, 1]
        @type gt_bboxes: Tensor
        @param gt_bboxes: Initial GT bboxes [bs, n_max_boxes, 4]
        @type pred_kpts: Tensor | None
        @param pred_kpts: Predicted keypoints [bs, n_anchors, n_kpts, 3]
            (optional)
        @type gt_kpts: Tensor | None
        @param gt_kpts: Ground truth keypoints [bs, n_max_boxes, n_kpts,
            3] (optional)
        @type sigmas: Tensor | None
        @param sigmas: Sigmas for OKS computation if keypoints are used.
            (optional)
        @type area_factor: float | None
        @param area_factor: Area factor for OKS computation if keypoints
            are used.
        @rtype: tuple[Tensor, Tensor]
        @return: Alignment metric and IoU between GTs and predicted
            bboxes, optionally incorporating pose OKS.
        """
        pred_scores = pred_scores.permute(0, 2, 1)
        gt_labels = gt_labels.to(torch.long)
        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)
        ind[0] = (
            torch.arange(end=self.bs).view(-1, 1).repeat(1, self.n_max_boxes)
        )
        ind[1] = gt_labels.squeeze(-1)
        bbox_scores = pred_scores[ind[0], ind[1]]

        overlaps = batch_iou(gt_bboxes, pred_bboxes)
        if all_not_none([pred_kpts, gt_kpts, sigmas, area_factor]):
            pose_oks = compute_pose_oks(
                pred_kpts,  # type: ignore
                gt_kpts,  # type: ignore
                sigmas=sigmas,  # type: ignore
                gt_bboxes=gt_bboxes,
                pose_area=None,
                eps=self.eps,
                area_factor=area_factor,  # type: ignore
                use_cocoeval_oks=True,
            )
            overlaps = overlaps * pose_oks

        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)

        return align_metric, overlaps

    def _select_topk_candidates(
        self,
        metrics: Tensor,
        largest: bool = True,
        topk_mask: Tensor | None = None,
    ) -> Tensor:
        """Selects k anchors based on provided metrics tensor.

        @type metrics: Tensor
        @param metrics: Metrics tensor of shape [bs, n_max_boxes,
            n_anchors]
        @type largest: bool
        @param largest: Flag if should keep largest topK. Defaults to
            True.
        @type topk_mask: Tensor
        @param topk_mask: Mask for valid GTs of shape [bs, n_max_boxes,
            topk]
        @rtype: Tensor
        @return: Mask of selected anchors of shape [bs, n_max_boxes,
            n_anchors]
        """
        n_anchors = metrics.shape[-1]
        topk_metrics, topk_idxs = torch.topk(
            metrics, self.topk, dim=-1, largest=largest
        )
        if topk_mask is None:
            topk_mask = (
                topk_metrics.max(dim=-1, keepdim=True)[0] > self.eps
            ).tile([1, 1, self.topk])
        topk_idxs = torch.where(
            topk_mask, topk_idxs, torch.zeros_like(topk_idxs)
        )
        is_in_topk = F.one_hot(topk_idxs, n_anchors).sum(dim=-2)
        is_in_topk = torch.where(
            is_in_topk > 1, torch.zeros_like(is_in_topk), is_in_topk
        )
        return is_in_topk.to(metrics.dtype)

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
        batch_ind = torch.arange(
            end=self.bs, dtype=torch.int64, device=gt_labels.device
        )[..., None]
        assigned_gt_idx = assigned_gt_idx + batch_ind * self.n_max_boxes
        assigned_labels = gt_labels.long().flatten()[assigned_gt_idx]

        # assigned target boxes
        assigned_bboxes = gt_bboxes.reshape([-1, 4])[assigned_gt_idx]

        # assigned target scores
        assigned_labels[assigned_labels < 0] = 0
        assigned_scores = F.one_hot(assigned_labels, self.n_classes)
        mask_pos_scores = mask_pos_sum[:, :, None].repeat(1, 1, self.n_classes)
        assigned_scores = torch.where(
            mask_pos_scores > 0,
            assigned_scores,
            torch.full_like(assigned_scores, 0),
        )

        assigned_labels = torch.where(
            mask_pos_sum.bool(),
            assigned_labels,
            torch.full_like(assigned_labels, self.n_classes),
        )
        return assigned_labels, assigned_bboxes, assigned_scores
