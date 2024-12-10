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
        reg_max: int = 16,
        tal_topk: int = 10,
        class_loss_weight: float = 0.5,
        bbox_loss_weight: float = 7.5,
        dfl_loss_weight: float = 1.5,
        overlap_mask: bool = True,
        **kwargs: Any,
    ):
        """Instance Segmentation and BBox loss adapted from  U{Real-Time Flying Object Detection with YOLOv8
        <https://arxiv.org/pdf/2305.09972>}

        @type reg_max: int
        @param reg_max: Maximum number of regression channels. Defaults to 16.
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
            reg_max=reg_max,
            tal_topk=tal_topk,
            class_loss_weight=class_loss_weight,
            bbox_loss_weight=bbox_loss_weight,
            dfl_loss_weight=dfl_loss_weight,
            **kwargs,
        )
        self.overlap = overlap_mask

    def prepare(
        self, inputs: Packet[Tensor], labels: Labels
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        det_feats = self.get_input_tensors(inputs, "features")
        proto = self.get_input_tensors(inputs, "prototypes")
        pred_mask = self.get_input_tensors(inputs, "mask_coeficients")
        self._init_parameters(det_feats)
        self.batch_size, _, mask_h, mask_w = proto.shape
        pred_distri, pred_scores = torch.cat(
            [xi.view(self.batch_size, self.node.no, -1) for xi in det_feats], 2
        ).split((self.node.reg_max * 4, self.n_classes), 1)
        target_bbox = self.get_label(labels, TaskType.BOUNDINGBOX)
        img_idx = target_bbox[:, 0]
        target_masks = self.get_label(labels, TaskType.INSTANCE_SEGMENTATION)
        if tuple(target_masks.shape[-2:]) != (mask_h, mask_w):
            target_masks = F.interpolate(
                target_masks.unsqueeze(0), (mask_h, mask_w), mode="nearest"
            ).squeeze(0)

        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_mask = pred_mask.permute(0, 2, 1).contiguous()

        target_bbox = self._preprocess_bbox_target(
            target_bbox, self.batch_size
        )

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
        max_assigned_scores_sum = max(assigned_scores.sum(), 1)
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

        loss_seg = self.calculate_segmentation_loss(
            mask_positive,
            target_masks,
            assigned_gt_idx,
            assigned_bboxes,
            img_idx,
            proto,
            pred_masks,
            self.overlap,
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

    # TODO: Modify after adding corect annotation loading
    def calculate_segmentation_loss(
        self,
        fg_mask: torch.Tensor,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        target_bboxes: torch.Tensor,
        batch_idx: torch.Tensor,
        proto: torch.Tensor,
        pred_masks: torch.Tensor,
        overlap: bool,
    ) -> torch.Tensor:
        """Calculate the loss for instance segmentation.

        Args:
            fg_mask (torch.Tensor): A binary tensor of shape (BS, N_anchors) indicating which anchors are positive.
            masks (torch.Tensor): Ground truth masks of shape (BS, H, W) if `overlap` is False, otherwise (BS, ?, H, W).
            target_gt_idx (torch.Tensor): Indexes of ground truth objects for each anchor of shape (BS, N_anchors).
            target_bboxes (torch.Tensor): Ground truth bounding boxes for each anchor of shape (BS, N_anchors, 4).
            batch_idx (torch.Tensor): Batch indices of shape (N_labels_in_batch, 1).
            proto (torch.Tensor): Prototype masks of shape (BS, 32, H, W).
            pred_masks (torch.Tensor): Predicted masks for each anchor of shape (BS, N_anchors, 32).
            imgsz (torch.Tensor): Size of the input image as a tensor of shape (2), i.e., (H, W).
            overlap (bool): Whether the masks in `masks` tensor overlap.

        Returns:
            (torch.Tensor): The calculated loss for instance segmentation.

        Notes:
            The batch loss can be computed for improved speed at higher memory usage.
            For example, pred_mask can be computed as follows:
                pred_mask = torch.einsum('in,nhw->ihw', pred, proto)  # (i, 32) @ (32, 160, 160) -> (i, 160, 160)
        """
        _, _, mask_h, mask_w = proto.shape
        loss = 0

        # Normalize to 0-1
        target_bboxes_normalized = target_bboxes / self.gt_bboxes_scale

        # Areas of target bboxes
        marea = box_convert(
            target_bboxes_normalized, in_fmt="xyxy", out_fmt="xywh"
        )[..., 2:].prod(2)

        # Normalize to mask size
        mxyxy = target_bboxes_normalized * torch.tensor(
            [mask_w, mask_h, mask_w, mask_h], device=proto.device
        )

        for i, single_i in enumerate(
            zip(fg_mask, target_gt_idx, pred_masks, proto, mxyxy, marea, masks)
        ):
            (
                fg_mask_i,
                target_gt_idx_i,
                pred_masks_i,
                proto_i,
                mxyxy_i,
                marea_i,
                masks_i,
            ) = single_i
            if fg_mask_i.any():
                mask_idx = target_gt_idx_i[fg_mask_i]
                if overlap:
                    gt_mask = masks_i == (mask_idx + 1).view(-1, 1, 1)
                    gt_mask = gt_mask.float()
                else:
                    gt_mask = masks[batch_idx.view(-1) == i][mask_idx]

                loss += self.single_mask_loss(
                    gt_mask,
                    pred_masks_i[fg_mask_i],
                    proto_i,
                    mxyxy_i[fg_mask_i],
                    marea_i[fg_mask_i],
                )

            # WARNING: lines below prevents Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
            else:
                loss += (proto * 0).sum() + (
                    pred_masks * 0
                ).sum()  # inf sums may lead to nan loss

        return loss / fg_mask.sum()

    # TODO: Modify after adding corect annotation loading
    @staticmethod
    def single_mask_loss(
        gt_mask: torch.Tensor,
        pred: torch.Tensor,
        proto: torch.Tensor,
        xyxy: torch.Tensor,
        area: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the instance segmentation loss for a single image.

        Args:
            gt_mask (torch.Tensor): Ground truth mask of shape (n, H, W), where n is the number of objects.
            pred (torch.Tensor): Predicted mask coefficients of shape (n, 32).
            proto (torch.Tensor): Prototype masks of shape (32, H, W).
            xyxy (torch.Tensor): Ground truth bounding boxes in xyxy format, normalized to [0, 1], of shape (n, 4).
            area (torch.Tensor): Area of each ground truth bounding box of shape (n,).

        Returns:
            (torch.Tensor): The calculated mask loss for a single image.

        Notes:
            The function uses the equation pred_mask = torch.einsum('in,nhw->ihw', pred, proto) to produce the
            predicted masks from the prototype masks and predicted mask coefficients.
        """
        pred_mask = torch.einsum(
            "in,nhw->ihw", pred, proto
        )  # (n, 32) @ (32, 80, 80) -> (n, 80, 80)
        loss = F.binary_cross_entropy_with_logits(
            pred_mask, gt_mask, reduction="none"
        )
        return (
            apply_bounding_box_to_masks(loss, xyxy).mean(dim=(1, 2)) / area
        ).sum()
