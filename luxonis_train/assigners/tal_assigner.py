"""The task-aligned assigner, which selects the positive anchors by the
predicted class score and the IoU of the predicted box together.
"""

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from loguru import logger
from luxonis_ml.typing import all_not_none, any_not_none
from torch import Tensor, nn

from luxonis_train.utils import compute_pose_oks

from .utils import batch_iou, candidates_in_gt, fix_collisions


class TaskAlignedAssigner(nn.Module):
    r"""Task-aligned assigner (TAL) from TOOD.

    The assigner scores each anchor for each ground truth box with the
    *alignment metric*:

    .. math::

        t = s^{\alpha} \cdot u^{\beta}

    Here, :math:`s` is the predicted score of the class of the box. The
    overlap :math:`u` is the IoU between the predicted box of the anchor
    and the ground truth box. With keypoints, :math:`u` is this IoU times
    the object keypoint similarity from `compute_pose_oks`. For each
    ground truth box, the assigner does these steps:

    - It finds the anchors that have a center inside the box.
    - It keeps at most ``topk`` of these anchors, the ones with the
      highest :math:`t`.

    An anchor that stays positive for more than one box goes to the box
    with the highest :math:`u`. For details, see
    `luxonis_train.assigners.utils.fix_collisions`. The assigned score of
    a positive anchor is :math:`t`, normalized for each box:

    .. math::

        \hat{t} = \frac{t}{\max t} \cdot \max u

    Both maxima are over the positive anchors of the box.

    *Small-Target-Aware Label Assignment* (STAL) helps small objects
    get positive anchors. When STAL is on, a side of a real box that is
    shorter than the smallest stride gets the length of the second
    smallest stride. With only one stride, it gets the length of that
    stride. The center of the box stays. The assigner uses the enlarged
    box only to find the anchors inside the box.

    `AdaptiveDetectionLoss` uses this assigner after its warmup epochs.
    `PrecisionDFLDetectionLoss` uses it in all epochs.

    References:
        - `TOOD: Task-aligned One-stage Object Detection
          <https://arxiv.org/pdf/2108.07755.pdf>`_
        - The implementation adapts the code of `PPYOLOE_pytorch
          <https://github.com/Nioolek/PPYOLOE_pytorch/blob/master/ppyoloe/assigner/tal_assigner.py>`_,
          which has the `Apache License, Version 2.0
          <https://github.com/Nioolek/PPYOLOE_pytorch/tree/master?tab=Apache-2.0-1-ov-file#readme>`_.

    """

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
        r"""Initialize the task-aligned assigner.

        Args:
            n_classes (int): Number of classes in the dataset. The label
                ``n_classes`` marks a background anchor in the output.
            topk (int): Number of anchors with the highest alignment
                metric that the assigner selects for each ground truth
                box. It must not be larger than the number of anchors.
            alpha (float): The exponent :math:`\alpha` of the class score
                in the alignment metric.
            beta (float): The exponent :math:`\beta` of the overlap
                :math:`u` in the alignment metric.
            eps (float): A small value that prevents a division by zero
                in the score normalization and in the object keypoint
                similarity.
            strides (``Sequence[int] | Tensor | None``): The strides of
                the detection head in pixels, for example ``[8, 16, 32]``.
                The assigner sorts them and removes duplicates. STAL
                needs the strides, so ``None`` or an empty value turns
                STAL off.
            skip_stal (bool): ``True`` turns STAL off. When
                ``skip_stal`` is ``False`` and ``strides`` is ``None`` or
                empty, the assigner logs a warning and turns STAL off.

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
        r"""Assign each anchor to a ground truth box or to the background.

        All boxes are in ``xyxy`` format. The boxes, the anchor points,
        and the keypoints use the same units. STAL compares the box sizes
        with the strides, so STAL needs the boxes in pixels. Only the
        boxes with ``mask_gt`` set to ``1`` get positive anchors. The
        method runs under `torch.no_grad`, so the outputs have no
        gradient.

        To add the object keypoint similarity to the alignment metric,
        give all four of ``pred_kpts``, ``gt_kpts``, ``sigmas``, and
        ``area_factor``.

        Args:
            pred_scores (``Tensor``): Predicted class probabilities in
                ``[0, 1]`` with shape ``[bs, n_anchors, n_classes]``. The
                losses give the sigmoid of the class logits.
            pred_bboxes (``Tensor``): Predicted boxes with shape
                ``[bs, n_anchors, 4]``.
            anchor_points (``Tensor``): Anchor centers ``(x, y)`` with
                shape ``[n_anchors, 2]``.
            gt_labels (``Tensor``): Class index of each ground truth box
                with shape ``[bs, n_max_boxes, 1]``.
            gt_bboxes (``Tensor``): Ground truth boxes with shape
                ``[bs, n_max_boxes, 4]``.
            mask_gt (``Tensor``): ``1`` for a real box and ``0`` for a
                padded slot, with shape ``[bs, n_max_boxes, 1]``.
            pred_kpts (``Tensor | None``): Predicted keypoints with shape
                ``[bs, n_anchors, n_kpts, 3]``. The assigner reads only
                ``x`` and ``y``.
            gt_kpts (``Tensor | None``): Ground truth keypoints as
                ``(x, y, visibility)`` with shape
                ``[bs, n_max_boxes, n_kpts, 3]``.
            sigmas (``Tensor | None``): One sigma per keypoint with shape
                ``[n_kpts]``.
            area_factor (float | None): The factor that scales the area of
                a ground truth box to the pose area.

        Returns:
            ``tuple[Tensor, Tensor, Tensor, Tensor, Tensor]``: Five tensors:

            - ``assigned_labels`` (``[bs, n_anchors]``, ``int64``): The
              class of the assigned box, or ``n_classes`` for a
              background anchor.
            - ``assigned_bboxes`` (``[bs, n_anchors, 4]``): The assigned
              box. Only the values at positive anchors are meaningful.
            - ``assigned_scores`` (``[bs, n_anchors, n_classes]``): A
              one-hot class vector scaled by the normalized alignment
              metric :math:`\hat{t}`. Zero for a background anchor.
            - ``mask_positive`` (``[bs, n_anchors]``, ``bool``): ``True``
              at an anchor with an assigned box.
            - ``assigned_gt_idx`` (``[bs, n_anchors]``, ``int64``): The
              index of the assigned box along dimension ``1`` of
              ``gt_bboxes``, ``0`` for a background anchor.

            When ``n_max_boxes`` is ``0``, every anchor is background.

        Raises:
            ValueError: When some, but not all, of ``pred_kpts``,
                ``gt_kpts``, ``sigmas``, and ``area_factor`` are ``None``.

        Notes:
            The call stores ``bs`` and ``n_max_boxes`` on the module.

        Example:
            The box holds the centers of the first two anchors. Both
            predicted boxes have an IoU of ``0.5`` with it. The first one
            has the higher class score, so it gets the score ``0.5``.

            >>> import torch
            >>> assigner = TaskAlignedAssigner(
            ...     n_classes=2, topk=2, skip_stal=True
            ... )
            >>> anchor_points = torch.tensor(
            ...     [[2.0, 2.0], [6.0, 2.0], [2.0, 6.0], [6.0, 6.0]]
            ... )
            >>> pred_bboxes = torch.cat(
            ...     [anchor_points - 2, anchor_points + 2], -1
            ... )
            >>> pred_scores = torch.tensor(
            ...     [[[0.1, 0.9], [0.2, 0.5], [0.3, 0.3], [0.9, 0.1]]]
            ... )
            >>> gt_labels = torch.tensor([[[1.0]]])
            >>> gt_bboxes = torch.tensor([[[0.0, 0.0, 8.0, 4.0]]])
            >>> mask_gt = torch.tensor([[[1.0]]])
            >>> labels, bboxes, scores, mask, gt_idx = assigner(
            ...     pred_scores,
            ...     pred_bboxes[None],
            ...     anchor_points,
            ...     gt_labels,
            ...     gt_bboxes,
            ...     mask_gt,
            ... )
            >>> labels.tolist()
            [[1, 1, 2, 2]]
            >>> mask.tolist()
            [[True, True, False, False]]
            >>> [round(s, 2) for s in scores[0, :, 1].tolist()]
            [0.5, 0.28, 0.0, 0.0]

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
        """Convert the strides to a sorted tuple of unique integers.

        Args:
            strides (``Sequence[int] | Tensor | None``): The strides of the
                detection head.

        Returns:
            ``tuple[int, ...] | None``: The sorted unique strides. ``None``
            when ``strides`` is ``None``. An empty ``strides`` gives an
            empty tuple.

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
        r"""Compute the alignment metric and the overlap.

        The method computes both values for each pair of a ground truth
        box and an anchor. The overlap :math:`u` is the IoU between the
        ground truth box and the predicted box. When no keypoint argument
        is ``None``, the overlap is the IoU times the object keypoint
        similarity. The metric is :math:`s^{\alpha} \cdot u^{\beta}`,
        where :math:`s` is the predicted score of the class of the box.

        Args:
            pred_scores (``Tensor``): Predicted class probabilities in
                ``[0, 1]`` with shape ``[bs, n_anchors, n_classes]``.
            pred_bboxes (``Tensor``): Predicted boxes with shape
                ``[bs, n_anchors, 4]``.
            gt_labels (``Tensor``): Class index of each ground truth box
                with shape ``[bs, n_max_boxes, 1]``.
            gt_bboxes (``Tensor``): Ground truth boxes with shape
                ``[bs, n_max_boxes, 4]``.
            pred_kpts (``Tensor | None``): Predicted keypoints with shape
                ``[bs, n_anchors, n_kpts, 3]``.
            gt_kpts (``Tensor | None``): Ground truth keypoints with shape
                ``[bs, n_max_boxes, n_kpts, 3]``.
            sigmas (``Tensor | None``): One sigma per keypoint with shape
                ``[n_kpts]``.
            area_factor (float | None): The factor that scales the area of
                a ground truth box to the pose area.

        Returns:
            ``tuple[Tensor, Tensor]``: The alignment metric and the
            overlap, both with shape ``[bs, n_max_boxes, n_anchors]``.

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
        """Mark the anchors that have a center inside each box.

        When STAL is on, the method first enlarges the small boxes with
        ``_expand_small_gt_bboxes``.

        Args:
            anchor_points (``Tensor``): Anchor centers ``(x, y)`` with
                shape ``[n_anchors, 2]``.
            gt_bboxes (``Tensor``): Ground truth boxes with shape
                ``[bs, n_max_boxes, 4]``.
            mask_gt (``Tensor``): ``1`` for a real box and ``0`` for a
                padded slot, with shape ``[bs, n_max_boxes, 1]``.

        Returns:
            ``Tensor``: Mask with shape ``[bs, n_max_boxes, n_anchors]``.
            ``1`` marks an anchor with a center inside the box.

        """
        if not self.skip_stal:
            gt_bboxes = self._expand_small_gt_bboxes(gt_bboxes, mask_gt)
        is_in_gts = candidates_in_gt(anchor_points, gt_bboxes.reshape(-1, 4))
        return is_in_gts.reshape(self.bs, self.n_max_boxes, -1)

    def _expand_small_gt_bboxes(
        self, gt_bboxes: Tensor, mask_gt: Tensor
    ) -> Tensor:
        """Enlarge the small ground truth boxes for STAL.

        A side of a real box that is shorter than ``min_stride`` gets the
        length ``stal_target_size``. The width and the height change
        independently. The center of the box stays.

        Args:
            gt_bboxes (``Tensor``): Ground truth boxes with shape
                ``[bs, n_max_boxes, 4]``.
            mask_gt (``Tensor``): ``1`` for a real box and ``0`` for a
                padded slot, with shape ``[bs, n_max_boxes, 1]``.

        Returns:
            ``Tensor``: The boxes with shape ``[bs, n_max_boxes, 4]``.
            ``gt_bboxes`` itself when ``min_stride`` or
            ``stal_target_size`` is ``None``.

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
        """Mark the ``topk`` anchors with the best metric for each box.

        The method sets each selected index that has a ``False`` value in
        ``topk_mask`` to ``0``, the first anchor, before it builds the
        mask. The mask does not mark an anchor that the method selects
        more than once for a box.

        Args:
            metrics (``Tensor``): The metric of each box and anchor with
                shape ``[bs, n_max_boxes, n_anchors]``.
            largest (bool): If ``True``, select the largest values. If
                ``False``, select the smallest values.
            topk_mask (``Tensor | None``): Boolean mask with shape
                ``[bs, n_max_boxes, topk]``. ``None`` keeps the selection
                of a box only when its largest selected metric is larger
                than ``eps``.

        Returns:
            ``Tensor``: Mask with shape ``[bs, n_max_boxes, n_anchors]``
            and the dtype of ``metrics``.

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
        """Gather the label, box, and one-hot score of each anchor.

        Before the one-hot encoding, the method changes a negative label
        to ``0``.

        Args:
            gt_labels (``Tensor``): Class index of each ground truth box
                with shape ``[bs, n_max_boxes, 1]``.
            gt_bboxes (``Tensor``): Ground truth boxes with shape
                ``[bs, n_max_boxes, 4]``.
            assigned_gt_idx (``Tensor``): Index of the assigned box with
                shape ``[bs, n_anchors]``.
            mask_pos_sum (``Tensor``): Number of assigned boxes per anchor
                with shape ``[bs, n_anchors]``.

        Returns:
            ``tuple[Tensor, Tensor, Tensor]``: Three tensors:

            - The ``int64`` assigned labels with shape ``[bs, n_anchors]``.
              ``n_classes`` for a background anchor.
            - The assigned boxes with shape ``[bs, n_anchors, 4]``.
            - The ``int64`` one-hot scores with shape
              ``[bs, n_anchors, n_classes]``. Zero for a background
              anchor.

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
