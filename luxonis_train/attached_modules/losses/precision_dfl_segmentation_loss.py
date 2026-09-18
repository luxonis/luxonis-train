"""The YOLOv8 detection loss with a mask term for instance
segmentation.
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from torchvision.ops import box_convert

from luxonis_train.nodes import PrecisionSegmentBBoxHead
from luxonis_train.tasks import Tasks
from luxonis_train.utils import apply_bounding_box_to_masks

from .precision_dfl_detection_loss import PrecisionDFLDetectionLoss


class PrecisionDFLSegmentationLoss(PrecisionDFLDetectionLoss):
    r"""Instance segmentation loss for `PrecisionSegmentBBoxHead`.

    The loss adds a mask term to `PrecisionDFLDetectionLoss`. For each
    positive anchor, the loss builds mask logits from the mask
    coefficients of the anchor and the prototypes of the head. A binary
    cross-entropy compares these logits with the target mask of the
    assigned box, inside that box only.

    Inputs:
        - ``features`` (``list[Tensor]``): :math:`\left[B, 4 * reg_max +
          n_{classes}, H_i, W_i\right]` per scale, the distance bin
          logits followed by the class logits
        - ``prototypes`` (``Tensor``): :math:`\left[B, n_{masks}, 2 *
          H_0, 2 * W_0\right]`
        - ``mask_coefficients`` (``Tensor``): :math:`\left[B, n_{masks},
          N\right]`, for the :math:`N` anchors of all scales
        - ``target_boundingbox`` (``Tensor``): :math:`\left[N_{gt},
          6\right]`, ``[batch_index, class, x, y, w, h]``, ``xywh``
          normalized, with ``x`` and ``y`` at the top-left corner
        - ``target_instance_segmentation`` (``Tensor``):
          :math:`\left[N_{gt}, H, W\right]`, one mask for each target
          box, in the same order

    Outputs:
        - ``Tensor``: scalar total loss
        - ``dict[str, Tensor]``: scalar sub-losses ``class``, ``iou``,
          ``dfl``, ``seg``, detached and without the weights

    Formula:
        The terms :math:`L_{cls}`, :math:`L_{iou}`, and :math:`L_{dfl}`
        are those of `PrecisionDFLDetectionLoss`. :math:`P` is the set of
        positive anchors in all images of the batch. For anchor
        :math:`a`, the mask logit map :math:`m_a = \sum_k c_{a,k} \, p_k`
        is the sum of the prototypes :math:`p_k`, weighted by the mask
        coefficients :math:`c_{a,k}` of the anchor. The prototypes have
        the size :math:`h \times w`. :math:`g_a` is the target mask of the
        assigned box, at the size :math:`h \times w`. :math:`A_a` is the
        area of that box as a fraction of the image area.
        :math:`\text{crop}_a` sets the pixels outside the box to ``0``:

        .. math::

            L_{seg} = \frac{1}{|P|} \sum_{a \in P} \frac{1}{A_a \, h w}
            \sum_{x, y} \text{crop}_a\left(
            \text{BCE}\left(m_a, g_a\right)\right)_{x, y}

            L = \lambda_{cls} L_{cls} + \lambda_{box} L_{iou}
            + \lambda_{dfl} L_{dfl} + \lambda_{box} L_{seg}

        The mask term uses the box weight ``bbox_loss_weight``.

    References:
        - Source: Reimplemented from `Real-Time Flying Object Detection
          with YOLOv8 <https://arxiv.org/abs/2305.09972>`_ and `YOLOv6:
          A Single-Stage Object Detection Framework for Industrial
          Applications <https://arxiv.org/abs/2209.02976>`_ and
          `PP-YOLOE: An evolved version of YOLO
          <https://arxiv.org/abs/2203.16250>`_.
        - License: Apache-2.0 (this project)

    Notes:
        The ``seg`` term and the total loss are **NaN** when the batch has
        no positive anchor. The mask term divides by the number of
        positive anchors, also when that number is ``0``. The notes of
        `PrecisionDFLDetectionLoss` on the anchor cache and on a
        ``reg_max`` of ``1`` apply here too.

    Example:
        Attached to a ``PrecisionSegmentBBoxHead`` in ``model.nodes``:

        .. code-block:: yaml

            - name: PrecisionSegmentBBoxHead
              inputs: [RepPANNeck]
              losses:
                - name: PrecisionDFLSegmentationLoss

    Compatible with:
        - Used by: `InstanceSegmentationModel`
        - Nodes: `PrecisionSegmentBBoxHead`

    """

    node: PrecisionSegmentBBoxHead
    supported_tasks = [Tasks.INSTANCE_SEGMENTATION]

    def __init__(
        self,
        tal_topk: int = 10,
        class_loss_weight: float = 0.5,
        bbox_loss_weight: float = 7.5,
        dfl_loss_weight: float = 1.5,
        skip_stal: bool = False,
        **kwargs,
    ):
        """Initialize the loss with the settings of the detection terms.

        The mask term has no settings of its own. The loss needs a
        ``node`` of type `PrecisionSegmentBBoxHead`. When a config uses
        the predefined `InstanceSegmentationModel` and
        ``trainer.smart_cfg_auto_populate`` is ``True``, the config
        changes the three weights. It sets each weight to its default
        times ``trainer.accumulate_grad_batches``. These values replace
        the weights in the ``loss_params`` of the model.

        Args:
            tal_topk (int): The ``topk`` of `TaskAlignedAssigner`, the
                largest number of positive anchors for each target box.
            class_loss_weight (float): Weight of the classification term.
            bbox_loss_weight (float): Weight of the CIoU box term and of
                the mask term.
            dfl_loss_weight (float): Weight of the DFL term.
            skip_stal (bool): Whether to turn off Small-Target-Aware Label
                Assignment (STAL) in the assigner.
            **kwargs (``Any``): Keyword arguments forwarded to
                `PrecisionDFLDetectionLoss`, such as ``node`` and
                ``final_loss_weight``.

        """
        super().__init__(
            tal_topk=tal_topk,
            class_loss_weight=class_loss_weight,
            bbox_loss_weight=bbox_loss_weight,
            dfl_loss_weight=dfl_loss_weight,
            skip_stal=skip_stal,
            **kwargs,
        )

    def forward(
        self,
        features: list[Tensor],
        prototypes: Tensor,
        mask_coefficients: Tensor,
        target_boundingbox: Tensor,
        target_instance_segmentation: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Compute the detection loss and the mask loss of one batch.

        First, the method prepares ``target_instance_segmentation``:

        - When it is empty, the method replaces it with an empty tensor
          of shape ``[0, h, w]``, the prototype size.
        - When its size is not the prototype size, the method resizes it
          to ``[h, w]`` with nearest interpolation.

        The class, box, and DFL terms are the same as in
        `PrecisionDFLDetectionLoss.forward`. The first call caches the
        anchor points and the image scale in the same way. The method
        also keeps the index of the assigned target box of each anchor.
        It passes this index and the assigned boxes in pixels to
        `compute_segmentation_loss`.

        Args:
            features (``list[Tensor]``): One tensor per scale, of shape
                ``[B, 4 * reg_max + n_classes, H_i, W_i]``. The
                ``features`` output of the node.
            prototypes (``Tensor``): Mask prototypes of shape
                ``[B, n_masks, h, w]``. The ``prototypes`` output of the
                node.
            mask_coefficients (``Tensor``): Mask coefficients of shape
                ``[B, n_masks, N]``, for the ``N`` anchors of all scales.
                The ``mask_coefficients`` output of the node.
            target_boundingbox (``Tensor``): Target boxes of shape
                ``[N_gt, 6]``, with rows ``[batch_index, class, x, y, w,
                h]``. The coordinates are ``xywh`` normalized to
                ``[0, 1]``, with ``x`` and ``y`` at the top-left corner.
                The ``boundingbox`` label of the task.
            target_instance_segmentation (``Tensor``): Target masks of
                shape ``[N_gt, H, W]``, one for each row of
                ``target_boundingbox``, in the same order. The
                ``instance_segmentation`` label of the task.

        Returns:
            ``tuple[Tensor, dict[str, Tensor]]``: The scalar weighted total
            loss, and a dictionary that maps ``"class"``, ``"iou"``,
            ``"dfl"``, and ``"seg"`` to the detached terms before the
            weights. The total loss and ``"seg"`` are ``NaN`` when no
            anchor is positive. The total loss and ``"dfl"`` have the
            shape ``[1]`` when ``reg_max`` of the node is ``1`` and an
            anchor is positive.

        """
        self._init_parameters(features)
        batch_size, _, mask_h, mask_w = prototypes.shape
        pred_distri, pred_scores = torch.cat(
            [xi.view(batch_size, self.node.no, -1) for xi in features], 2
        ).split((self.node.reg_max * 4, self.n_classes), 1)
        img_idx = target_boundingbox[:, 0].unsqueeze(-1)
        if target_instance_segmentation.numel() == 0:
            target_instance_segmentation = torch.empty(
                (0, mask_h, mask_w),
                device=target_instance_segmentation.device,
                dtype=target_instance_segmentation.dtype,
            )
        elif tuple(target_instance_segmentation.shape[-2:]) != (
            mask_h,
            mask_w,
        ):
            target_instance_segmentation = F.interpolate(
                target_instance_segmentation.unsqueeze(0),
                (mask_h, mask_w),
                mode="nearest",
            ).squeeze(0)

        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        mask_coefficients = mask_coefficients.permute(0, 2, 1).contiguous()

        target_boundingbox = self._preprocess_bbox_target(
            target_boundingbox, batch_size
        )

        pred_bboxes = self.decode_bbox(
            self._anchor_points_strided, pred_distri
        )

        gt_labels = target_boundingbox[:, :, :1]
        gt_xyxy = target_boundingbox[:, :, 1:]
        mask_gt = (gt_xyxy.sum(-1, keepdim=True) > 0).float()

        _, assigned_bboxes, assigned_scores, mask_positive, assigned_gt_idx = (
            self.assigner(
                pred_scores.detach().sigmoid(),
                (pred_bboxes.detach() * self._stride_tensor).type(
                    gt_xyxy.dtype
                ),
                self._anchor_points,
                gt_labels,
                gt_xyxy,
                mask_gt,
            )
        )

        max_assigned_scores_sum = max(assigned_scores.sum().item(), 1)
        loss_cls = (
            self.bce(pred_scores, assigned_scores)
        ).sum() / max_assigned_scores_sum
        if mask_positive.sum():
            loss_iou, loss_dfl = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                self._anchor_points_strided,
                assigned_bboxes / self._stride_tensor,
                assigned_scores,
                max_assigned_scores_sum,
                mask_positive,
            )
        else:
            loss_iou = torch.tensor(0.0).to(pred_distri.device)
            loss_dfl = torch.tensor(0.0).to(pred_distri.device)

        loss_seg = self.compute_segmentation_loss(
            mask_positive,
            target_instance_segmentation,
            assigned_gt_idx,
            assigned_bboxes,
            img_idx,
            prototypes,
            mask_coefficients,
        )

        loss = (
            self._class_loss_weight * loss_cls
            + self._bbox_loss_weight * loss_iou
            + self._dfl_loss_weight * loss_dfl
            + self._bbox_loss_weight * loss_seg
        )
        sub_losses = {
            "class": loss_cls.detach(),
            "iou": loss_iou.detach(),
            "dfl": loss_dfl.detach(),
            "seg": loss_seg.detach(),
        }

        return loss, sub_losses

    def compute_segmentation_loss(
        self,
        fg_mask: Tensor,
        gt_masks: Tensor,
        gt_idx: Tensor,
        bboxes: Tensor,
        batch_ids: Tensor,
        proto: Tensor,
        pred_masks: Tensor,
    ) -> Tensor:
        """Compute the mask term of the whole batch.

        For each positive anchor of an image, the method multiplies the
        mask coefficients with the prototypes into mask logits. It
        computes the binary cross-entropy between these logits and the
        target mask of the assigned box. It sets the loss outside the box
        to ``0``. The loss of the anchor is the mean over all mask pixels.
        The method divides this mean by the area of the box as a fraction
        of the image area. An image with no
        positive anchor adds a zero that depends on ``proto`` and
        ``pred_masks``, so both stay in the autograd graph. The method
        reads the image scale that `forward` caches, so `forward` must
        run first.

        Args:
            fg_mask (``Tensor``): Boolean mask of the positive anchors, of
                shape ``[B, N]``.
            gt_masks (``Tensor``): Target masks of all images, of shape
                ``[N_gt, h, w]``, at the prototype size. The masks of one
                image are in the order of its target boxes.
            gt_idx (``Tensor``): For each anchor, the index of the
                assigned box among the target boxes of its image, of
                shape ``[B, N]``.
            bboxes (``Tensor``): Assigned boxes in ``xyxy`` pixels of the
                input image, of shape ``[B, N, 4]``.
            batch_ids (``Tensor``): Image index of each target mask, of
                shape ``[N_gt, 1]``.
            proto (``Tensor``): Mask prototypes of shape
                ``[B, n_masks, h, w]``.
            pred_masks (``Tensor``): Mask coefficients of shape
                ``[B, N, n_masks]``.

        Returns:
            ``Tensor``: The scalar sum of the anchor losses, divided by the
            number of positive anchors. ``NaN`` when ``fg_mask`` has no
            positive anchor.

        """
        _, _, h, w = proto.shape
        total_loss = 0
        bboxes_norm = bboxes / self.gt_bboxes_scale
        bbox_area = box_convert(bboxes_norm, in_fmt="xyxy", out_fmt="xywh")[
            ..., 2:
        ].prod(2)
        bboxes_scaled = bboxes_norm * torch.tensor(
            [w, h, w, h], device=proto.device
        )

        for img_idx, data in enumerate(
            zip(
                fg_mask,
                gt_idx,
                pred_masks,
                proto,
                bboxes_scaled,
                bbox_area,
                strict=True,
            )
        ):
            fg, gt, pred, pr, bbox, area = data
            if fg.any():
                mask_ids = gt[fg]
                gt_mask = gt_masks[batch_ids.view(-1) == img_idx][mask_ids]

                # Compute individual image mask loss
                pred_mask = torch.einsum("in,nhw->ihw", pred[fg], pr)
                loss = F.binary_cross_entropy_with_logits(
                    pred_mask, gt_mask, reduction="none"
                )
                total_loss += (
                    apply_bounding_box_to_masks(loss, bbox[fg]).mean(
                        dim=(1, 2)
                    )
                    / area[fg]
                ).sum()
            else:
                total_loss += (proto * 0).sum() + (pred_masks * 0).sum()

        return total_loss / fg_mask.sum()
