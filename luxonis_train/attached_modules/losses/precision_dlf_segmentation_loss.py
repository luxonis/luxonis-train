import logging
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from torchvision.ops import box_convert

from luxonis_train.attached_modules.losses.precision_dfl_detection_loss import (
    PrecisionDFLDetectionLoss,
)
from luxonis_train.enums import TaskType
from luxonis_train.nodes import PrecisionSegmentBBoxHead
from luxonis_train.utils import (
    Labels,
    Packet,
    apply_bounding_box_to_masks,
)

logger = logging.getLogger(__name__)


class PrecisionDFLSegmentationLoss(PrecisionDFLDetectionLoss):
    node: PrecisionSegmentBBoxHead
    supported_tasks: list[TaskType] = [
        TaskType.BOUNDINGBOX,
        TaskType.INSTANCE_SEGMENTATION,
    ]

    def __init__(
        self,
        tal_topk: int = 10,
        class_loss_weight: float = 0.5,
        bbox_loss_weight: float = 7.5,
        dfl_loss_weight: float = 1.5,
        **kwargs: Any,
    ):
        """Instance Segmentation and BBox loss adapted from  U{Real-Time Flying Object Detection with YOLOv8
        <https://arxiv.org/pdf/2305.09972>} and from U{YOLOv6: A Single-Stage Object Detection Framework for Industrial Applications
        <https://arxiv.org/pdf/2209.02976.pdf>}.
        Code is adapted from U{https://github.com/Nioolek/PPYOLOE_pytorch/blob/master/ppyoloe/models}.

        @type tal_topk: int
        @param tal_topk: Number of anchors considered in selection. Defaults to 10.
        @type class_loss_weight: float
        @param class_loss_weight: Weight for classification loss. Defaults to 0.5. For optimal results, multiply with accumulate_grad_batches.
        @type bbox_loss_weight: float
        @param bbox_loss_weight: Weight for bbox loss. Defaults to 7.5. For optimal results, multiply with accumulate_grad_batches.
        @type dfl_loss_weight: float
        @param dfl_loss_weight: Weight for DFL loss. Defaults to 1.5. For optimal results, multiply with accumulate_grad_batches.
        """
        super().__init__(
            tal_topk=tal_topk,
            class_loss_weight=class_loss_weight,
            bbox_loss_weight=bbox_loss_weight,
            dfl_loss_weight=dfl_loss_weight,
            **kwargs,
        )

    def prepare(
        self, inputs: Packet[Tensor], labels: Labels
    ) -> tuple[
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
    ]:
        det_feats = self.get_input_tensors(inputs, "features")
        proto = self.get_input_tensors(inputs, "prototypes")[0]
        pred_mask = self.get_input_tensors(inputs, "mask_coeficients")[0]
        self._init_parameters(det_feats)
        batch_size, _, mask_h, mask_w = proto.shape
        pred_distri, pred_scores = torch.cat(
            [xi.view(batch_size, self.node.no, -1) for xi in det_feats], 2
        ).split((self.node.reg_max * 4, self.n_classes), 1)
        target_bbox = self.get_label(labels, TaskType.BOUNDINGBOX)
        img_idx = target_bbox[:, 0].unsqueeze(-1)
        target_masks = self.get_label(labels, TaskType.INSTANCE_SEGMENTATION)
        if tuple(target_masks.shape[-2:]) != (mask_h, mask_w):
            target_masks = F.interpolate(
                target_masks.unsqueeze(0), (mask_h, mask_w), mode="nearest"
            ).squeeze(0)

        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_mask = pred_mask.permute(0, 2, 1).contiguous()

        target_bbox = self._preprocess_bbox_target(target_bbox, batch_size)

        pred_bboxes = self.decode_bbox(self.anchor_points_strided, pred_distri)

        gt_labels = target_bbox[:, :, :1]
        gt_xyxy = target_bbox[:, :, 1:]
        mask_gt = (gt_xyxy.sum(-1, keepdim=True) > 0).float()

        _, assigned_bboxes, assigned_scores, mask_positive, assigned_gt_idx = (
            self.assigner(
                pred_scores.detach().sigmoid(),
                (pred_bboxes.detach() * self.stride_tensor).type(
                    gt_xyxy.dtype
                ),
                self.anchor_points,
                gt_labels,
                gt_xyxy,
                mask_gt,
            )
        )

        return (
            pred_distri,
            pred_bboxes,
            pred_scores,
            assigned_bboxes,
            assigned_scores,
            mask_positive,
            assigned_gt_idx,
            pred_mask,
            proto,
            target_masks,
            img_idx,
        )

    def forward(
        self,
        pred_distri: Tensor,
        pred_bboxes: Tensor,
        pred_scores: Tensor,
        assigned_bboxes: Tensor,
        assigned_scores: Tensor,
        mask_positive: Tensor,
        assigned_gt_idx: Tensor,
        pred_masks: Tensor,
        proto: Tensor,
        target_masks: Tensor,
        img_idx: Tensor,
    ):
        max_assigned_scores_sum = max(assigned_scores.sum().item(), 1)
        loss_cls = (
            self.bce(pred_scores, assigned_scores)
        ).sum() / max_assigned_scores_sum
        if mask_positive.sum():
            loss_iou, loss_dfl = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                self.anchor_points_strided,
                assigned_bboxes / self.stride_tensor,
                assigned_scores,
                max_assigned_scores_sum,
                mask_positive,
            )
        else:
            loss_iou = torch.tensor(0.0).to(pred_distri.device)
            loss_dfl = torch.tensor(0.0).to(pred_distri.device)

        loss_seg = self.compute_segmentation_loss(
            mask_positive,
            target_masks,
            assigned_gt_idx,
            assigned_bboxes,
            img_idx,
            proto,
            pred_masks,
        )

        loss = (
            self.class_loss_weight * loss_cls
            + self.bbox_loss_weight * loss_iou
            + self.dfl_loss_weight * loss_dfl
            + self.bbox_loss_weight * loss_seg
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
        fg_mask: torch.Tensor,
        gt_masks: torch.Tensor,
        gt_idx: torch.Tensor,
        bboxes: torch.Tensor,
        batch_ids: torch.Tensor,
        proto: torch.Tensor,
        pred_masks: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the segmentation loss for the entire batch.

        @type fg_mask: torch.Tensor
        @param fg_mask: Foreground mask. Shape: (B, N_anchor).
        @type gt_masks: torch.Tensor
        @param gt_masks: Ground truth masks. Shape: (n, H, W).
        @type gt_idx: torch.Tensor
        @param gt_idx: Ground truth mask indices. Shape: (B, N_anchor).
        @type bboxes: torch.Tensor
        @param bboxes: Ground truth bounding boxes in xyxy format.
            Shape: (B, N_anchor, 4).
        @type batch_ids: torch.Tensor
        @param batch_ids: Batch indices. Shape: (n, 1).
        @type proto: torch.Tensor
        @param proto: Prototype masks. Shape: (B, 32, H, W).
        @type pred_masks: torch.Tensor
        @param pred_masks: Predicted mask coefficients. Shape: (B,
            N_anchor, 32).
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
            zip(fg_mask, gt_idx, pred_masks, proto, bboxes_scaled, bbox_area)
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
