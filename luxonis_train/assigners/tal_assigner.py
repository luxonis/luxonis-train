from collections.abc import Sequence

import torch
import torch.nn.functional as F
from loguru import logger
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
        strides: Sequence[int] | Tensor | None = None,
        skip_stal: bool = False,
    ):
        """Task Aligned Assigner.

        Adapted from `TOOD: Task-aligned One-stage Object Detection
        <https://arxiv.org/pdf/2108.07755.pdf>`_. Code is adapted from
        `PPYOLOE_pytorch <https://github.com/Nioolek/PPYOLOE_pytorch/blob/master/ppyoloe/assigner/tal_assigner.py>`_.

        The adapted implementation is distributed under the `Apache License,
        Version 2.0
        <https://github.com/Nioolek/PPYOLOE_pytorch/tree/master?tab=Apache-2.0-1-ov-file#readme>`_.

        Args:
            n_classes (int): Number of classes in the dataset.
            topk (int): Number of anchors considered in selection.
                Defaults to ``13``.
            alpha (float): Classification score exponent. Defaults to
                ``1.0``.
            beta (float): Localization overlap exponent. Defaults to
                ``6.0``.
            eps (float): Small value used to avoid division by zero.
                Defaults to ``1e-9``.
            strides (Sequence[int] | Tensor | None): Detection strides,
                usually ``8``, ``16``, and ``32``.
            skip_stal (bool): If ``True``, disables Small-Target-Aware Label
                Assignment candidate expansion.
        """
        super().__init__()

        self.n_classes = n_classes
        self.topk = topk
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        normalized_strides = self._normalize_strides(strides)
        self.strides = normalized_strides or None
        if not skip_stal and self.strides is None:
            logger.warning(
                "STAL was requested for TaskAlignedAssigner, but no valid "
                "`strides` were provided. `strides` should be the detection "
                "head stride values in pixels, for example `[8, 16, 32]`. "
                "nodes inheriting from `BaseDetectionHead` provide this "
                "attribute. "
            )
        self.skip_stal = bool(skip_stal or not self.strides)
        self.min_stride = self.strides[0] if self.strides is not None else None
        self.stal_target_size = (
            self.strides[1]
            if self.strides is not None and len(self.strides) > 1
            else self.min_stride
        )

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
        """Generate final assignments.

        If both pred_kpts and gt_kpts are provided, a pose OKS is
        computed and used in the alignment metric; the final tuple then
        includes assigned poses.

        Args:
            pred_scores (Tensor): Predicted scores with shape
                ``[bs, n_anchors, 1]``.
            pred_bboxes (Tensor): Predicted bboxes with shape
                ``[bs, n_anchors, 4]``.
            anchor_points (Tensor): Anchor points with shape
                ``[n_anchors, 2]``.
            gt_labels (Tensor): Initial GT labels with shape
                ``[bs, n_max_boxes, 1]``.
            gt_bboxes (Tensor): Initial GT bboxes with shape
                ``[bs, n_max_boxes, 4]``.
            mask_gt (Tensor): Mask for valid GTs with shape
                ``[bs, n_max_boxes, 1]``.
            pred_kpts (Tensor | None): Optional predicted keypoints with
                shape ``[bs, n_anchors, n_kpts, 3]``.
            gt_kpts (Tensor | None): Optional ground truth keypoints with
                shape ``[bs, n_max_boxes, n_kpts, 3]``.
            sigmas (Tensor | None): Sigmas for OKS computation if keypoints
                are used.
            area_factor (float | None): Area factor for OKS computation.

        Returns:
            tuple[Tensor, Tensor, Tensor, Tensor, Tensor]: Assigned labels
            with shape ``[bs, n_anchors]``, assigned bboxes with shape
            ``[bs, n_anchors, 4]``, assigned scores with shape
            ``[bs, n_anchors, n_classes]``, output mask with shape
            ``[bs, n_anchors]``, and assigned GT indices with shape
            ``[bs, n_anchors]``.

        Raises:
            ValueError: If only some of ``pred_kpts``, ``gt_kpts``,
                ``sigmas``, and ``area_factor`` are provided.
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
        is_in_gts = self._select_candidates_in_gts(
            anchor_points, gt_bboxes, mask_gt
        )
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

    def _normalize_strides(
        self, strides: Sequence[int] | Tensor | None
    ) -> tuple[int, ...] | None:
        """Normalize stride values to a sorted tuple.

        Args:
            strides (Sequence[int] | Tensor | None): Detection stride values.

        Returns:
            tuple[int, ...] | None: Sorted unique integer stride values, or
            ``None`` when no strides are provided.
        """
        if strides is None:
            return None

        if isinstance(strides, Tensor):
            strides = strides.detach().cpu().tolist()

        return tuple(sorted({int(stride) for stride in strides}))

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
        """Calculate anchor alignment metric and IoU between GTs and
        predicted bboxes (optionally incorporating pose OKS).

        Args:
            pred_scores (Tensor): Predicted scores with shape
                ``[bs, n_anchors, 1]``.
            pred_bboxes (Tensor): Predicted bboxes with shape
                ``[bs, n_anchors, 4]``.
            gt_labels (Tensor): Initial GT labels with shape
                ``[bs, n_max_boxes, 1]``.
            gt_bboxes (Tensor): Initial GT bboxes with shape
                ``[bs, n_max_boxes, 4]``.
            pred_kpts (Tensor | None): Optional predicted keypoints with
                shape ``[bs, n_anchors, n_kpts, 3]``.
            gt_kpts (Tensor | None): Optional ground truth keypoints with
                shape ``[bs, n_max_boxes, n_kpts, 3]``.
            sigmas (Tensor | None): Optional sigmas for OKS computation.
            area_factor (float | None): Optional area factor for OKS
                computation.

        Returns:
            tuple[Tensor, Tensor]: Alignment metric and IoU between GTs and
            predicted bboxes, optionally incorporating pose OKS.
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

    def _select_candidates_in_gts(
        self, anchor_points: Tensor, gt_bboxes: Tensor, mask_gt: Tensor
    ) -> Tensor:
        """Select anchors whose centers lie inside ground truth boxes.

        Args:
            anchor_points (Tensor): Anchor points with shape
                ``[n_anchors, 2]``.
            gt_bboxes (Tensor): Ground truth bboxes with shape
                ``[bs, n_max_boxes, 4]``.
            mask_gt (Tensor): Mask for valid GTs with shape
                ``[bs, n_max_boxes, 1]``.

        Returns:
            Tensor: Candidate mask with shape
            ``[bs, n_max_boxes, n_anchors]``.
        """
        if not self.skip_stal:
            gt_bboxes = self._expand_small_gt_bboxes(gt_bboxes, mask_gt)
        is_in_gts = candidates_in_gt(anchor_points, gt_bboxes.reshape(-1, 4))
        return is_in_gts.reshape(self.bs, self.n_max_boxes, -1)

    def _expand_small_gt_bboxes(
        self, gt_bboxes: Tensor, mask_gt: Tensor
    ) -> Tensor:
        """Expand small ground truth boxes for STAL candidate selection.

        Args:
            gt_bboxes (Tensor): Ground truth bboxes with shape
                ``[bs, n_max_boxes, 4]``.
            mask_gt (Tensor): Mask for valid GTs with shape
                ``[bs, n_max_boxes, 1]``.

        Returns:
            Tensor: Possibly expanded GT bboxes with shape
            ``[bs, n_max_boxes, 4]``.
        """
        if self.min_stride is None or self.stal_target_size is None:
            return gt_bboxes

        gt_centers = (gt_bboxes[..., :2] + gt_bboxes[..., 2:]) / 2
        gt_wh = (gt_bboxes[..., 2:] - gt_bboxes[..., :2]).clamp_min(0)
        small_mask = (gt_wh < self.min_stride) & mask_gt.bool()
        expanded_wh = torch.where(
            small_mask,
            torch.full_like(gt_wh, float(self.stal_target_size)),
            gt_wh,
        )
        half_wh = expanded_wh / 2
        return torch.cat((gt_centers - half_wh, gt_centers + half_wh), dim=-1)

    def _select_topk_candidates(
        self,
        metrics: Tensor,
        largest: bool = True,
        topk_mask: Tensor | None = None,
    ) -> Tensor:
        """Select k anchors based on provided metrics tensor.

        Args:
            metrics (Tensor): Metrics tensor with shape
                ``[bs, n_max_boxes, n_anchors]``.
            largest (bool): Whether to keep the largest top-k values.
                Defaults to ``True``.
            topk_mask (Tensor | None): Optional mask for valid GTs with shape
                ``[bs, n_max_boxes, topk]``.

        Returns:
            Tensor: Mask of selected anchors with shape
            ``[bs, n_max_boxes, n_anchors]``.
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
