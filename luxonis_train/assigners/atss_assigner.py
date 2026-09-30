"""Adaptive Training Sample Selection, which selects the positive
anchors of each ground truth box by an adaptive IoU threshold.
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .utils import batch_iou, bbox_iou, candidates_in_gt, fix_collisions


class ATSSAssigner(nn.Module):
    r"""Adaptive Training Sample Selection (ATSS) assigner.

    The assigner selects the positive anchors of each ground truth box
    from the geometry of the anchors. It reads the predicted boxes only
    to scale the assigned scores. For each ground truth box, it does
    these steps:

    - On each pyramid level, it selects up to ``topk`` anchors that have
      the centers closest to the center of the box. These anchors are
      the *candidates*.
    - It computes the IoU between each candidate and the box. The
      threshold of the box is :math:`\mu + \sigma`, the mean plus the
      standard deviation of these IoUs.
    - It keeps the candidates that have an IoU above the threshold and a
      center inside the box.

    An anchor that stays positive for more than one box goes to the box
    that has the highest IoU with the anchor box. For details, see
    `luxonis_train.assigners.utils.fix_collisions`.

    `AdaptiveDetectionLoss` uses this assigner for the first
    ``n_warmup_epochs`` epochs. After these epochs, it uses
    `TaskAlignedAssigner`.

    References:
        - `Bridging the Gap Between Anchor-based and Anchor-free
          Detection via Adaptive Training Sample Selection
          <https://arxiv.org/pdf/1912.02424.pdf>`_
        - The implementation adapts the code of `PPYOLOE_pytorch
          <https://github.com/Nioolek/PPYOLOE_pytorch/blob/master/ppyoloe/assigner/atss_assigner.py>`_
          and `TOOD
          <https://github.com/fcjian/TOOD/blob/master/mmdet/core/bbox/assigners/atss_assigner.py>`_.

    """

    def __init__(self, n_classes: int, topk: int = 9):
        """Initialize the ATSS assigner.

        Args:
            n_classes (int): Number of classes in the dataset. The label
                ``n_classes`` marks a background anchor in the output.
            topk (int): Maximum number of candidate anchors to select on
                each pyramid level for each ground truth box. With fewer
                than three candidates for a box over all levels, no
                candidate passes the threshold.

        """
        super().__init__()

        self._topk = topk
        self._n_classes = n_classes

    def forward(
        self,
        anchor_bboxes: Tensor,
        n_level_bboxes: list[int],
        gt_labels: Tensor,
        gt_bboxes: Tensor,
        mask_gt: Tensor,
        pred_bboxes: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Assign each anchor to a ground truth box or to the background.

        All boxes are in ``xyxy`` format and in the same units. Only the
        boxes with ``mask_gt`` set to ``1`` get positive anchors.

        Args:
            anchor_bboxes (``Tensor``): Anchor boxes with shape
                ``[n_anchors, 4]``, ordered level by level.
            n_level_bboxes (list[int]): Number of anchors on each pyramid
                level. The sum must equal ``n_anchors``.
            gt_labels (``Tensor``): Class index of each ground truth box
                with shape ``[bs, n_max_boxes, 1]``.
            gt_bboxes (``Tensor``): Ground truth boxes with shape
                ``[bs, n_max_boxes, 4]``.
            mask_gt (``Tensor``): ``1`` for a real box and ``0`` for a
                padded slot, with shape ``[bs, n_max_boxes, 1]``.
            pred_bboxes (``Tensor``): Predicted boxes with shape
                ``[bs, n_anchors, 4]``. The IoU between a predicted box
                and its assigned box scales the assigned scores.

        Returns:
            ``tuple[Tensor, Tensor, Tensor, Tensor, Tensor]``: Five tensors:

            - ``assigned_labels`` (``[bs, n_anchors]``, ``int64``): The
              class of the assigned box, or ``n_classes`` for a
              background anchor.
            - ``assigned_bboxes`` (``[bs, n_anchors, 4]``): The assigned
              box. Only the values at positive anchors are meaningful.
            - ``assigned_scores`` (``[bs, n_anchors, n_classes]``): A
              one-hot class vector scaled by the IoU between the
              predicted box and the assigned box. Zero for a background
              anchor.
            - ``mask_positive`` (``[bs, n_anchors]``, ``bool``): ``True``
              at an anchor with an assigned box.
            - ``assigned_gt_idx`` (``[bs, n_anchors]``, ``int64``): The
              index of the assigned box along dimension ``1`` of
              ``gt_bboxes``, ``0`` for a background anchor.

            When ``n_max_boxes`` is ``0``, every anchor is background. In
            this case, ``mask_positive`` and ``assigned_gt_idx`` are
            ``float32`` zeros.

        Example:
            The box has an IoU of ``0.64`` with the first anchor box. The
            threshold is about ``0.5``, so only the first anchor is
            positive. The predicted boxes equal the anchor boxes, so the
            score of that anchor is also ``0.64``.

            >>> import torch
            >>> assigner = ATSSAssigner(n_classes=2, topk=4)
            >>> anchors = torch.tensor(
            ...     [
            ...         [0.0, 0.0, 4.0, 4.0],
            ...         [4.0, 0.0, 8.0, 4.0],
            ...         [0.0, 4.0, 4.0, 8.0],
            ...         [4.0, 4.0, 8.0, 8.0],
            ...     ]
            ... )
            >>> gt_labels = torch.tensor([[[1.0]]])
            >>> gt_bboxes = torch.tensor([[[0.0, 0.0, 5.0, 5.0]]])
            >>> mask_gt = torch.tensor([[[1.0]]])
            >>> labels, bboxes, scores, mask, gt_idx = assigner(
            ...     anchors, [4], gt_labels, gt_bboxes, mask_gt, anchors[None]
            ... )
            >>> labels.tolist()
            [[1, 2, 2, 2]]
            >>> mask.tolist()
            [[True, False, False, False]]
            >>> [round(s, 2) for s in scores[0, 0].tolist()]
            [0.0, 0.64]

        """
        self._n_anchors = anchor_bboxes.size(0)
        self._bs = gt_bboxes.size(0)
        self._n_max_boxes = gt_bboxes.size(1)

        if self._n_max_boxes == 0:
            device = gt_bboxes.device
            return (
                torch.full([self._bs, self._n_anchors], self._n_classes).to(
                    device
                ),
                torch.zeros([self._bs, self._n_anchors, 4]).to(device),
                torch.zeros([self._bs, self._n_anchors, self._n_classes]).to(
                    device
                ),
                torch.zeros([self._bs, self._n_anchors]).to(device),
                torch.zeros([self._bs, self._n_anchors]).to(device),
            )

        gt_bboxes_flat = gt_bboxes.reshape([-1, 4])

        # Compute iou between all gt and anchor bboxes
        overlaps = bbox_iou(gt_bboxes_flat, anchor_bboxes)
        overlaps = overlaps.reshape([self._bs, -1, self._n_anchors])

        # Compute center distance between all gt and anchor bboxes
        gt_centers = self._get_bbox_center(gt_bboxes_flat)
        anchor_centers = self._get_bbox_center(anchor_bboxes)
        distances = (
            (gt_centers[:, None, :] - anchor_centers[None, :, :])
            .pow(2)
            .sum(-1)
            .sqrt()
        )
        distances = distances.reshape([self._bs, -1, self._n_anchors])

        # Select candidates based on the center distance
        is_in_topk, topk_idxs = self._select_topk_candidates(
            distances, n_level_bboxes, mask_gt
        )

        # Compute threshold and selected positive candidates based on it
        is_pos = self._get_positive_samples(is_in_topk, topk_idxs, overlaps)

        # Select candidates inside GT
        is_in_gts = candidates_in_gt(anchor_centers, gt_bboxes_flat)
        is_in_gts = is_in_gts.reshape(self._bs, self._n_max_boxes, -1)

        # Final positive candidates
        mask_pos = is_pos * is_in_gts * mask_gt

        # If an anchor box is assigned to multiple gts, the one with the highest IoU is selected
        assigned_gt_idx, mask_pos_sum, mask_pos = fix_collisions(
            mask_pos, overlaps, self._n_max_boxes
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
        """Compute the center of each box.

        Args:
            bbox (``Tensor``): Boxes in ``xyxy`` format with shape
                ``[N, 4]``.

        Returns:
            ``Tensor``: Centers ``(x, y)`` with shape ``[N, 2]``.

        """
        cx = (bbox[:, 0] + bbox[:, 2]) / 2.0
        cy = (bbox[:, 1] + bbox[:, 3]) / 2.0
        return torch.stack((cx, cy), dim=1).to(bbox.device)

    def _select_topk_candidates(
        self, distances: Tensor, n_level_bboxes: list[int], mask_gt: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Select up to ``topk`` closest anchors on each level.

        For a padded slot, the method sets each selected index to ``0``,
        the first anchor of the level, before it builds the mask. The
        mask does not mark an anchor that the method selects more than
        once for a box. ``topk_idxs`` keeps the indices from before this
        change.

        Args:
            distances (``Tensor``): Distances between the box centers and
                the anchor centers with shape
                ``[bs, n_max_boxes, n_anchors]``.
            n_level_bboxes (list[int]): Number of anchors on each pyramid
                level.
            mask_gt (``Tensor``): ``1`` for a real box and ``0`` for a
                padded slot, with shape ``[bs, n_max_boxes, 1]``.

        Returns:
            ``tuple[Tensor, Tensor]``: The mask ``is_in_topk`` with shape
            ``[bs, n_max_boxes, n_anchors]`` and the indices ``topk_idxs``
            with shape ``[bs, n_max_boxes, n_selected]``. ``n_selected``
            is the number of candidates over all levels. Each index points
            into all ``n_anchors`` anchors, not into one level.

        """
        mask_gt = mask_gt.bool()
        level_distances = distances.split(n_level_bboxes, dim=-1)
        is_in_topk_list: list[Tensor] = []
        topk_idxs: list[Tensor] = []
        start_idx = 0
        for per_level_distances, per_level_boxes in zip(
            level_distances, n_level_bboxes, strict=True
        ):
            end_idx = start_idx + per_level_boxes
            selected_k = min(self._topk, per_level_boxes)
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
        """Keep the candidates with an IoU above the threshold.

        The threshold of a box is the mean plus the standard deviation of
        the IoUs of its candidates.

        Args:
            is_in_topk (``Tensor``): Candidate mask with shape
                ``[bs, n_max_boxes, n_anchors]``.
            topk_idxs (``Tensor``): Candidate indices with shape
                ``[bs, n_max_boxes, n_selected]``.
            overlaps (``Tensor``): IoU between each box and each anchor
                with shape ``[bs, n_max_boxes, n_anchors]``.

        Returns:
            ``Tensor``: Positive mask with shape
            ``[bs, n_max_boxes, n_anchors]``.

        """
        n_bs_max_boxes = self._bs * self._n_max_boxes
        _candidate_overlaps = torch.where(
            is_in_topk > 0, overlaps, torch.zeros_like(overlaps)
        )
        topk_idxs = topk_idxs.reshape([n_bs_max_boxes, -1])
        assist_idxs = self._n_anchors * torch.arange(
            n_bs_max_boxes, device=topk_idxs.device
        )
        assist_idxs = assist_idxs[:, None]
        flatten_idxs = topk_idxs + assist_idxs
        candidate_overlaps = _candidate_overlaps.reshape(-1)[flatten_idxs]
        candidate_overlaps = candidate_overlaps.reshape(
            [self._bs, self._n_max_boxes, -1]
        )

        overlaps_mean_per_gt = candidate_overlaps.mean(dim=-1, keepdim=True)
        overlaps_std_per_gt = candidate_overlaps.std(dim=-1, keepdim=True)
        overlaps_threshold_per_gt = overlaps_mean_per_gt + overlaps_std_per_gt

        return torch.where(
            _candidate_overlaps
            > overlaps_threshold_per_gt.repeat([1, 1, self._n_anchors]),
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
        """Gather the label, box, and one-hot score of each anchor.

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

            - The assigned labels with shape ``[bs, n_anchors]`` and the
              dtype of ``gt_labels``. ``n_classes`` for a background
              anchor.
            - The assigned boxes with shape ``[bs, n_anchors, 4]``.
            - The ``float32`` one-hot scores with shape
              ``[bs, n_anchors, n_classes]``. Zero for a background
              anchor.

        """
        # assigned target labels
        batch_idx = torch.arange(
            self._bs, dtype=gt_labels.dtype, device=gt_labels.device
        )
        batch_idx = batch_idx[..., None]
        assigned_gt_idx = (
            assigned_gt_idx + batch_idx * self._n_max_boxes
        ).long()
        assigned_labels = gt_labels.flatten()[assigned_gt_idx.flatten()]
        assigned_labels = assigned_labels.reshape([self._bs, self._n_anchors])
        assigned_labels = torch.where(
            mask_pos_sum > 0,
            assigned_labels,
            torch.full_like(assigned_labels, self._n_classes),
        )

        # assigned target boxes
        assigned_bboxes = gt_bboxes.reshape([-1, 4])[assigned_gt_idx.flatten()]
        assigned_bboxes = assigned_bboxes.reshape(
            [self._bs, self._n_anchors, 4]
        )

        # assigned target scores
        assigned_scores = F.one_hot(
            assigned_labels.long(), self._n_classes + 1
        ).float()
        assigned_scores = assigned_scores[:, :, : self._n_classes]

        return assigned_labels, assigned_bboxes, assigned_scores
