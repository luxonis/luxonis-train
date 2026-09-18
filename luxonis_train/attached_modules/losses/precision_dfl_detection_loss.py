"""The YOLOv8 detection loss: classification, box regression, and
distribution focal loss over the regression bins.
"""

from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.ops import box_convert

from luxonis_train.assigners import TaskAlignedAssigner
from luxonis_train.nodes import PrecisionBBoxHead
from luxonis_train.tasks import Tasks
from luxonis_train.utils import (
    anchors_for_fpn_features,
    bbox2dist,
    bbox_iou,
    dist2bbox,
)

from .base_loss import BaseLoss


class PrecisionDFLDetectionLoss(BaseLoss):
    r"""Bounding box loss for `PrecisionBBoxHead`, with a distribution
    focal loss on the box sides.

    The loss decodes a box for each anchor from the distance bins of the
    head. `TaskAlignedAssigner` then selects the positive anchors and
    gives each of them a target box and target class scores. The target
    class scores of the other anchors are ``0``. The loss has three
    terms:

    - a binary cross-entropy on the class logits of all anchors;
    - a CIoU loss on the boxes of the positive anchors;
    - a distribution focal loss (DFL) on the distance bins of the
      positive anchors.

    `BBoxLoss` computes the last two terms.

    Inputs:
        - ``features`` (``list[Tensor]``): :math:`\left[B, 4 * reg_max +
          n_{classes}, H_i, W_i\right]` per scale, the distance bin
          logits followed by the class logits
        - ``target`` (``Tensor``): :math:`\left[N_{gt}, 6\right]`,
          ``[batch_index, class, x, y, w, h]``, ``xywh`` normalized, with
          ``x`` and ``y`` at the top-left corner

    Outputs:
        - ``Tensor``: scalar total loss
        - ``dict[str, Tensor]``: scalar sub-losses ``class``, ``iou``,
          ``dfl``, detached and without the weights

    Formula:
        The sums run over the anchors of all scales in all images of the
        batch. For anchor :math:`a` and class :math:`c`, :math:`z_{a,c}`
        is the class logit and :math:`\hat{t}_{a,c}` the score from the
        assigner. :math:`P` is the set of positive anchors. :math:`b_a`
        is the predicted box and :math:`\hat{b}_a` the assigned box, both
        in units of the stride of the anchor. Each positive anchor has
        the weight :math:`w_a = \sum_c \hat{t}_{a,c}`, and the normalizer
        is :math:`S = \max\left(\sum_a w_a, 1\right)`:

        .. math::

            L_{cls} = \frac{1}{S} \sum_a \sum_c
            \text{BCE}\left(z_{a,c}, \hat{t}_{a,c}\right)

            L_{iou} = \frac{1}{S} \sum_{a \in P} w_a
            \left(1 - \text{CIoU}\left(b_a, \hat{b}_a\right)\right)

            L_{dfl} = \frac{1}{S} \sum_{a \in P} w_a \, \text{DFL}_a

            L = \lambda_{cls} L_{cls} + \lambda_{box} L_{iou}
            + \lambda_{dfl} L_{dfl}

        :math:`\text{DFL}_a` is the mean `DFLoss` of the four sides of
        the box. The weights :math:`\lambda` are ``class_loss_weight``,
        ``bbox_loss_weight``, and ``dfl_loss_weight``. :math:`L_{iou}`
        and :math:`L_{dfl}` are ``0`` when no anchor is positive.

    References:
        - Source: Reimplemented from `Real-Time Flying Object Detection
          with YOLOv8 <https://arxiv.org/abs/2305.09972>`_ and `YOLOv6:
          A Single-Stage Object Detection Framework for Industrial
          Applications <https://arxiv.org/abs/2209.02976>`_ and
          `PP-YOLOE: An evolved version of YOLO
          <https://arxiv.org/abs/2203.16250>`_.
        - License: Apache-2.0 (this project)

    Notes:
        The first call of ``forward`` caches the anchor points, their
        strides, and the image scale. It computes them from the shapes of
        ``features``, and from ``stride``, ``grid_cell_offset``, and
        ``original_in_shape`` of the node. Later calls reuse the cache,
        so every batch must have the feature map sizes of the first
        batch. The assigner works in pixels of the input image. The box
        and DFL terms work in units of the stride. When ``reg_max`` of
        the node is ``1``, the ``dfl`` term is always ``0``. It then has
        the shape ``[1]`` when an anchor is positive, and so does the
        total loss.

    Example:
        Attached to a ``PrecisionBBoxHead`` in ``model.nodes``:

        .. code-block:: yaml

            - name: PrecisionBBoxHead
              inputs: [RepPANNeck]
              losses:
                - name: PrecisionDFLDetectionLoss

    Compatible with:
        - Nodes: `PrecisionBBoxHead`

    """

    node: PrecisionBBoxHead
    supported_tasks = [Tasks.BOUNDINGBOX]

    def __init__(
        self,
        tal_topk: int = 10,
        class_loss_weight: float = 0.5,
        bbox_loss_weight: float = 7.5,
        dfl_loss_weight: float = 1.5,
        skip_stal: bool = False,
        **kwargs,
    ):
        """Initialize the assigner, the box loss, and the class loss.

        The loss reads ``n_classes``, ``original_in_shape``, ``stride``,
        ``grid_cell_size``, ``grid_cell_offset``, and ``reg_max`` from
        the node. Without a ``node``, the constructor raises
        ``RuntimeError``. The code adapts `PPYOLOE_pytorch
        <https://github.com/Nioolek/PPYOLOE_pytorch/blob/master/ppyoloe/models>`_.
        For the best results, multiply each weight by
        ``trainer.accumulate_grad_batches``.

        Args:
            tal_topk (int): The ``topk`` of `TaskAlignedAssigner`, the
                largest number of positive anchors for each target box.
                The exponents of the assigner are fixed: ``alpha`` is
                ``0.5`` and ``beta`` is ``6.0``.
            class_loss_weight (float): Weight of the classification term.
            bbox_loss_weight (float): Weight of the CIoU box term.
            dfl_loss_weight (float): Weight of the DFL term.
            skip_stal (bool): Whether to turn off Small-Target-Aware Label
                Assignment (STAL) in the assigner. When a side of a target
                box is shorter than the smallest stride, STAL gives that
                side the length of the second smallest stride. The
                assigner uses the enlarged box only to find the anchors
                inside the box.
            **kwargs (``Any``): Keyword arguments forwarded to `BaseLoss`,
                such as ``node`` and ``final_loss_weight``.

        """
        super().__init__(**kwargs)
        self._stride = self.node.stride
        self._grid_cell_size = self.node.grid_cell_size
        self._grid_cell_offset = self.node.grid_cell_offset
        self._original_img_size = self.original_in_shape[1:]

        self._class_loss_weight = class_loss_weight
        self._bbox_loss_weight = bbox_loss_weight
        self._dfl_loss_weight = dfl_loss_weight

        self.assigner = TaskAlignedAssigner(
            n_classes=self.n_classes,
            topk=tal_topk,
            alpha=0.5,
            beta=6.0,
            strides=self._stride,
            skip_stal=skip_stal,
        )
        self.bbox_loss = BBoxLoss(self.node.reg_max)
        self._proj = torch.arange(self.node.reg_max, dtype=torch.float)
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(
        self, features: list[Tensor], target: Tensor
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Compute the detection loss of one batch.

        The method splits ``features`` into the distance bin logits and
        the class logits of all anchors. It pads ``target`` to the same
        number of boxes for each image, and converts the boxes to
        ``xyxy`` pixels. `decode_bbox` gives the predicted boxes.
        `TaskAlignedAssigner` reads detached copies of the class
        probabilities and of the predicted boxes in pixels. A target box
        with a sum of its ``xyxy`` coordinates of ``0`` or less counts as
        padding. The first call caches the anchor points and the image
        scale, as the class notes describe.

        Args:
            features (``list[Tensor]``): One tensor per scale, of shape
                ``[B, 4 * reg_max + n_classes, H_i, W_i]``. The
                ``features`` output of the node.
            target (``Tensor``): Target boxes of shape ``[N_gt, 6]``, with
                rows ``[batch_index, class, x, y, w, h]``. The coordinates
                are ``xywh`` normalized to ``[0, 1]``, with ``x`` and
                ``y`` at the top-left corner. The ``boundingbox`` label of
                the task.

        Returns:
            ``tuple[Tensor, dict[str, Tensor]]``: The scalar weighted total
            loss, and a dictionary that maps ``"class"``, ``"iou"``, and
            ``"dfl"`` to the detached terms before the weights. The total
            loss and ``"dfl"`` have the shape ``[1]`` when ``reg_max`` of
            the node is ``1`` and an anchor is positive.

        """
        self._init_parameters(features)
        batch_size = features[0].shape[0]
        pred_distri, pred_scores = torch.cat(
            [xi.view(batch_size, self.node.no, -1) for xi in features], 2
        ).split((self.node.reg_max * 4, self.n_classes), 1)
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()

        target = self._preprocess_bbox_target(target, batch_size)

        pred_bboxes = self.decode_bbox(
            self._anchor_points_strided, pred_distri
        )

        gt_labels = target[:, :, :1]
        gt_xyxy = target[:, :, 1:]
        mask_gt = (gt_xyxy.sum(-1, keepdim=True) > 0).float()

        _, assigned_bboxes, assigned_scores, mask_positive, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * self._stride_tensor).type(gt_xyxy.dtype),
            self._anchor_points,
            gt_labels,
            gt_xyxy,
            mask_gt,
        )
        assigned_bboxes /= self._stride_tensor

        max_assigned_scores_sum = max(assigned_scores.sum().item(), 1)
        loss_cls = (
            self.bce(pred_scores, assigned_scores)
        ).sum() / max_assigned_scores_sum
        if mask_positive.sum():
            loss_iou, loss_dfl = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                self._anchor_points_strided,
                assigned_bboxes,
                assigned_scores,
                max_assigned_scores_sum,
                mask_positive,
            )
        else:
            loss_iou = torch.tensor(0.0).to(pred_distri.device)
            loss_dfl = torch.tensor(0.0).to(pred_distri.device)

        loss = (
            self._class_loss_weight * loss_cls
            + self._bbox_loss_weight * loss_iou
            + self._dfl_loss_weight * loss_dfl
        )
        sub_losses = {
            "class": loss_cls.detach(),
            "iou": loss_iou.detach(),
            "dfl": loss_dfl.detach(),
        }

        return loss, sub_losses

    def _preprocess_bbox_target(
        self, target: Tensor, batch_size: int
    ) -> Tensor:
        sample_ids, counts = cast(
            tuple[Tensor, Tensor],
            torch.unique(target[:, 0].int(), return_counts=True),
        )
        c_max = int(counts.max()) if counts.numel() > 0 else 0
        out_target = torch.zeros(batch_size, c_max, 5, device=target.device)
        out_target[:, :, 0] = -1
        for id, count in zip(sample_ids, counts, strict=True):
            out_target[id, :count] = target[target[:, 0] == id][:, 1:]

        scaled_target = out_target[:, :, 1:5] * self.gt_bboxes_scale
        out_target[..., 1:] = box_convert(scaled_target, "xywh", "xyxy")

        return out_target

    def decode_bbox(self, anchor_points: Tensor, pred_dist: Tensor) -> Tensor:
        """Decode the distance bin logits into boxes.

        For each side of each box, the method applies a softmax over the
        ``reg_max`` bins. The distance of the side is the expected bin
        index under these probabilities. `dist2bbox` then turns the four
        distances into a box around the anchor point. The method also
        decodes the bins when ``reg_max`` is ``1``, and then every
        distance is ``0``.

        Args:
            anchor_points (``Tensor``): Anchor centers ``(x, y)`` of shape
                ``[N, 2]``. `forward` passes them in units of the stride
                of each anchor.
            pred_dist (``Tensor``): Distance bin logits of shape
                ``[B, N, 4 * reg_max]``, with the sides in the order left,
                top, right, bottom.

        Returns:
            ``Tensor``: Boxes of shape ``[B, N, 4]`` in ``xyxy`` format, in
            the units of ``anchor_points``.

        """
        if self.node.dfl:
            batch_size, n_anchors, n_channels = pred_dist.shape
            dist_probs = pred_dist.view(
                batch_size, n_anchors, 4, n_channels // 4
            ).softmax(dim=3)
            dist_transformed = dist_probs @ self._proj.to(
                anchor_points.device, dtype=pred_dist.dtype
            )
        return dist2bbox(dist_transformed, anchor_points, out_format="xyxy")

    def _init_parameters(self, features: list[Tensor]) -> None:
        if not hasattr(self, "gt_bboxes_scale"):
            _, self._anchor_points, _, self._stride_tensor = (
                anchors_for_fpn_features(
                    features,
                    self._stride,
                    self._grid_cell_size,
                    self._grid_cell_offset,
                    multiply_with_stride=True,
                )
            )
            self.gt_bboxes_scale = torch.tensor(
                [
                    self._original_img_size[1],
                    self._original_img_size[0],
                    self._original_img_size[1],
                    self._original_img_size[0],
                ],
                device=features[0].device,
            )
            self._anchor_points_strided = (
                self._anchor_points / self._stride_tensor
            )


class BBoxLoss(nn.Module):
    r"""CIoU and distribution focal loss terms of the positive anchors.

    `PrecisionDFLDetectionLoss` and `PrecisionDFLSegmentationLoss` use it
    for their ``iou`` and ``dfl`` terms. :math:`P` is the set of positive
    anchors. Anchor :math:`a` has the predicted box :math:`b_a`, the
    target box :math:`\hat{b}_a`, and the weight :math:`w_a`, the sum of
    its target class scores. :math:`S` is the normalizer that `forward`
    receives as ``total_score``:

    .. math::

        L_{iou} = \frac{1}{S} \sum_{a \in P} w_a
        \left(1 - \text{CIoU}\left(b_a, \hat{b}_a\right)\right)

        L_{dfl} = \frac{1}{S} \sum_{a \in P} w_a \, \text{DFL}_a

    :math:`\text{DFL}_a` is the mean `DFLoss` over the four distances
    from the anchor point to the sides of the target box.

    """

    def __init__(self, reg_max: int = 16):
        """Initialize the loss and its DFL part.

        Args:
            reg_max (int): Number of distance bins for each side of a
                box. When ``reg_max`` is ``1`` or less, the loss has no
                DFL part.

        """
        super().__init__()
        self.dist_loss = DFLoss(reg_max) if reg_max > 1 else None

    def forward(
        self,
        pred_dist: Tensor,
        pred_bboxes: Tensor,
        anchors: Tensor,
        targets: Tensor,
        scores: Tensor,
        total_score: Tensor,
        fg_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        r"""Compute the CIoU term and the DFL term.

        ``pred_bboxes``, ``anchors``, and ``targets`` must use one unit,
        the size of one distance bin. The DFL term clips each target
        distance to ``[0, reg_max - 1.01]``. The CIoU term does not clip
        the boxes.

        Args:
            pred_dist (``Tensor``): Distance bin logits of shape
                ``[B, N, 4 * reg_max]``.
            pred_bboxes (``Tensor``): Predicted ``xyxy`` boxes of shape
                ``[B, N, 4]``, decoded from ``pred_dist``.
            anchors (``Tensor``): Anchor centers ``(x, y)`` of shape
                ``[N, 2]``.
            targets (``Tensor``): Assigned ``xyxy`` target boxes of shape
                ``[B, N, 4]``.
            scores (``Tensor``): Assigned class scores of shape
                ``[B, N, n_classes]``.
            total_score (``Tensor``): The normalizer :math:`S`. The
                detection losses pass the sum of ``scores``, at least
                ``1``, as a Python number.
            fg_mask (``Tensor``): Boolean mask of the positive anchors, of
                shape ``[B, N]``.

        Returns:
            ``tuple[Tensor, Tensor]``: The scalar CIoU term and the scalar
            DFL term. Without a DFL part, the DFL term is a zero tensor of
            shape ``[1]``.

        Example:
            The predicted box is equal to the target box, so the CIoU term
            is ``0``. Uniform bin logits give the DFL term :math:`\ln 4`:

            >>> import torch
            >>> loss = BBoxLoss(reg_max=4)
            >>> anchors = torch.tensor([[2.5, 2.5]])
            >>> boxes = torch.tensor([[[1.5, 1.5, 3.5, 3.5]]])
            >>> pred_dist = torch.zeros(1, 1, 16)
            >>> scores, total = torch.ones(1, 1, 1), torch.tensor(1.0)
            >>> fg_mask = torch.tensor([[True]])
            >>> iou, dfl = loss(
            ...     pred_dist, boxes, anchors, boxes, scores, total, fg_mask
            ... )
            >>> round(iou.item(), 4), round(dfl.item(), 4)
            (0.0, 1.3863)

        """
        score_weights = scores.sum(dim=-1)[fg_mask].unsqueeze(dim=-1)

        iou_vals = bbox_iou(
            pred_bboxes[fg_mask],
            targets[fg_mask],
            iou_type="ciou",
            element_wise=True,
        ).unsqueeze(dim=-1)
        iou_loss_val = ((1.0 - iou_vals) * score_weights).sum() / total_score

        if self.dist_loss is not None:
            offset_targets = bbox2dist(
                targets, anchors, self.dist_loss.reg_max - 1
            )
            dfl_loss_val = (
                self.dist_loss(
                    pred_dist[fg_mask].view(-1, self.dist_loss.reg_max),
                    offset_targets[fg_mask],
                )
                * score_weights
            )
            dfl_loss_val = dfl_loss_val.sum() / total_score
        else:
            dfl_loss_val = torch.zeros(1, device=pred_dist.device)

        return iou_loss_val, dfl_loss_val


class DFLoss(nn.Module):
    r"""Distribution focal loss (DFL) over the distance bins of a box
    side.

    A target distance :math:`y` is a real number between the bins
    :math:`y_l = \lfloor y \rfloor` and :math:`y_r = y_l + 1`. With the
    softmax probabilities :math:`p` over the bins, the loss is

    .. math::

        \text{DFL} = -\left(\left(y_r - y\right) \log p_{y_l}
        + \left(y - y_l\right) \log p_{y_r}\right)

    The loss is lowest when bin :math:`y_l` has the probability
    :math:`y_r - y` and bin :math:`y_r` has the probability
    :math:`y - y_l`.

    """

    def __init__(self, reg_max: int = 16):
        """Initialize the loss.

        Args:
            reg_max (int): Number of distance bins for each side of a
                box.

        """
        super().__init__()
        self.reg_max = reg_max

    def __call__(self, pred_dist: Tensor, targets: Tensor) -> Tensor:
        r"""Compute the DFL of each target box.

        The method clamps the target distances to
        ``[0, reg_max - 1.01]``, so that bin :math:`y_r` exists. The
        class overrides ``__call__`` and not ``forward``.

        Args:
            pred_dist (``Tensor``): Bin logits of shape
                ``[4 * M, reg_max]``, four rows for each box, with the
                sides in the order of ``targets``.
            targets (``Tensor``): Target distances of shape ``[M, 4]``, in
                units of one bin.

        Returns:
            ``Tensor``: The loss of each box, the mean over its four
            sides, of shape ``[M, 1]``.

        Examples:
            Uniform logits over four bins give :math:`\ln 4`:

            >>> import torch
            >>> loss = DFLoss(reg_max=4)
            >>> targets = torch.full((1, 4), 1.5)
            >>> loss(torch.zeros(4, 4), targets).shape
            torch.Size([1, 1])
            >>> round(loss(torch.zeros(4, 4), targets).item(), 4)
            1.3863

            For the distance ``1.5``, the best logits split the
            probability equally between bins ``1`` and ``2``. The loss is
            then :math:`\ln 2`:

            >>> logits = torch.tensor([[-100.0, 0.0, 0.0, -100.0]])
            >>> round(loss(logits.repeat(4, 1), targets).item(), 4)
            0.6931

        """
        targets = targets.clamp(0, self.reg_max - 1 - 0.01)
        left_target = targets.floor().long()
        right_target = left_target + 1
        weight_left = right_target - targets
        weight_right = 1.0 - weight_left

        left_val = F.cross_entropy(
            pred_dist, left_target.view(-1), reduction="none"
        ).view(left_target.shape)
        right_val = F.cross_entropy(
            pred_dist, right_target.view(-1), reduction="none"
        ).view(left_target.shape)

        return (left_val * weight_left + right_val * weight_right).mean(
            dim=-1, keepdim=True
        )
