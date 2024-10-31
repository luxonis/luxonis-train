import logging
from typing import Any, Literal, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.ops import box_convert

from luxonis_train.assigners import ATSSAssigner, TaskAlignedAssigner
from luxonis_train.enums import TaskType
from luxonis_train.nodes import EfficientBBoxHead
from luxonis_train.utils import (
    Labels,
    Packet,
    anchors_for_fpn_features,
    compute_iou_loss,
    dist2bbox,
)
from luxonis_train.utils.boundingbox import IoUType

from .base_loss import BaseLoss

logger = logging.getLogger(__name__)


class AdaptiveDetectionLoss(
    BaseLoss[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]
):
    node: EfficientBBoxHead
    supported_tasks: list[TaskType] = [TaskType.BOUNDINGBOX]

    anchors: Tensor
    anchor_points: Tensor
    n_anchors_list: list[int]
    stride_tensor: Tensor
    gt_bboxes_scale: Tensor

    def __init__(
        self,
        n_warmup_epochs: int = 4,
        iou_type: IoUType = "giou",
        reduction: Literal["sum", "mean"] = "mean",
        class_loss_weight: float = 1.0,
        iou_loss_weight: float = 2.5,
        **kwargs: Any,
    ):
        """BBox loss adapted from U{YOLOv6: A Single-Stage Object Detection Framework for Industrial Applications
        <https://arxiv.org/pdf/2209.02976.pdf>}. It combines IoU based bbox regression loss and varifocal loss
        for classification.
        Code is adapted from U{https://github.com/Nioolek/PPYOLOE_pytorch/blob/master/ppyoloe/models}.

        @type n_warmup_epochs: int
        @param n_warmup_epochs: Number of epochs where ATSS assigner is used, after that we switch to TAL assigner.
        @type iou_type: L{IoUType}
        @param iou_type: IoU type used for bbox regression loss.
        @type reduction: Literal["sum", "mean"]
        @param reduction: Reduction type for loss.
        @type class_loss_weight: float
        @param class_loss_weight: Weight of classification loss.
        @type iou_loss_weight: float
        @param iou_loss_weight: Weight of IoU loss.
        """
        super().__init__(**kwargs)

        self.iou_type: IoUType = iou_type
        self.reduction = reduction
        self.stride = self.node.stride
        self.grid_cell_size = self.node.grid_cell_size
        self.grid_cell_offset = self.node.grid_cell_offset
        self.original_img_size = self.original_in_shape[1:]

        self.n_warmup_epochs = n_warmup_epochs
        self.atss_assigner = ATSSAssigner(topk=9, n_classes=self.n_classes)
        self.tal_assigner = TaskAlignedAssigner(
            topk=13, n_classes=self.n_classes, alpha=1.0, beta=6.0
        )

        self.varifocal_loss = VarifocalLoss()
        self.class_loss_weight = class_loss_weight
        self.iou_loss_weight = iou_loss_weight

        self._logged_assigner_change = False

    def prepare(
        self, inputs: Packet[Tensor], labels: Labels
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        feats = self.get_input_tensors(inputs, "features")
        pred_scores = self.get_input_tensors(inputs, "class_scores")[0]
        pred_distri = self.get_input_tensors(inputs, "distributions")[0]

        target = self.get_label(labels)

        batch_size = pred_scores.shape[0]

        self._init_parameters(feats)

        target = self._preprocess_bbox_target(target, batch_size)
        pred_bboxes = dist2bbox(pred_distri, self.anchor_points_strided)

        gt_labels = target[:, :, :1]
        gt_xyxy = target[:, :, 1:]
        mask_gt = (gt_xyxy.sum(-1, keepdim=True) > 0).float()

        (
            assigned_labels,
            assigned_bboxes,
            assigned_scores,
            mask_positive,
            _,
        ) = self._run_assigner(
            gt_labels,
            gt_xyxy,
            mask_gt,
            pred_bboxes,
            pred_scores,
        )

        return (
            pred_bboxes,
            pred_scores,
            assigned_bboxes / self.stride_tensor,
            assigned_labels,
            assigned_scores,
            mask_positive,
        )

    def forward(
        self,
        pred_bboxes: Tensor,
        pred_scores: Tensor,
        assigned_bboxes: Tensor,
        assigned_labels: Tensor,
        assigned_scores: Tensor,
        mask_positive: Tensor,
    ):
        one_hot_label = F.one_hot(assigned_labels.long(), self.n_classes + 1)[
            ..., :-1
        ]
        loss_cls = self.varifocal_loss(
            pred_scores, assigned_scores, one_hot_label
        )

        if assigned_scores.sum() > 1:
            loss_cls /= assigned_scores.sum()

        loss_iou = compute_iou_loss(
            pred_bboxes,
            assigned_bboxes,
            assigned_scores,
            mask_positive,
            reduction="sum",
            iou_type=self.iou_type,
            bbox_format="xyxy",
        )[0]

        loss = (
            self.class_loss_weight * loss_cls + self.iou_loss_weight * loss_iou
        )

        sub_losses = {"class": loss_cls.detach(), "iou": loss_iou.detach()}

        return loss, sub_losses

    def _init_parameters(self, features: list[Tensor]):
        if not hasattr(self, "gt_bboxes_scale"):
            self.gt_bboxes_scale = torch.tensor(
                [
                    self.original_img_size[1],
                    self.original_img_size[0],
                    self.original_img_size[1],
                    self.original_img_size[0],
                ],
                device=features[0].device,
            )
            (
                self.anchors,
                self.anchor_points,
                self.n_anchors_list,
                self.stride_tensor,
            ) = anchors_for_fpn_features(
                features,
                self.stride,
                self.grid_cell_size,
                self.grid_cell_offset,
                multiply_with_stride=True,
            )
            self.anchor_points_strided = (
                self.anchor_points / self.stride_tensor
            )

    def _run_assigner(
        self,
        gt_labels: Tensor,
        gt_xyxy: Tensor,
        mask_gt: Tensor,
        pred_bboxes: Tensor,
        pred_scores: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        if self._epoch < self.n_warmup_epochs:
            return self.atss_assigner(
                self.anchors,
                self.n_anchors_list,
                gt_labels,
                gt_xyxy,
                mask_gt,
                pred_bboxes.detach() * self.stride_tensor,
            )
        else:
            self._log_assigner_change()
            return self.tal_assigner(
                pred_scores.detach(),
                pred_bboxes.detach() * self.stride_tensor,
                self.anchor_points,
                gt_labels,
                gt_xyxy,
                mask_gt,
            )

    def _preprocess_bbox_target(
        self, target: Tensor, batch_size: int
    ) -> Tensor:
        """Preprocess target in shape [batch_size, N, 5] where N is the
        maximum number of instances in one image."""
        sample_ids, counts = cast(
            tuple[Tensor, Tensor],
            torch.unique(target[:, 0].int(), return_counts=True),
        )
        c_max = int(counts.max()) if counts.numel() > 0 else 0
        out_target = torch.zeros(batch_size, c_max, 5, device=target.device)
        out_target[:, :, 0] = -1
        for id, count in zip(sample_ids, counts):
            out_target[id, :count] = target[target[:, 0] == id][:, 1:]

        scaled_target = out_target[:, :, 1:5] * self.gt_bboxes_scale
        out_target[..., 1:] = box_convert(scaled_target, "xywh", "xyxy")
        return out_target

    def _log_assigner_change(self):
        if self._logged_assigner_change:
            return

        logger.info(
            f"Switching to Task Aligned Assigner after {self.n_warmup_epochs} warmup epochs.",
            stacklevel=2,
        )
        self._logged_assigner_change = True


class VarifocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        """Varifocal Loss is a loss function for training a dense object detector to predict
        the IoU-aware classification score, inspired by focal loss.
        Code is adapted from: U{https://github.com/Nioolek/PPYOLOE_pytorch/blob/master/ppyoloe/models/losses.py}

        @type alpha: float
        @param alpha: alpha parameter in focal loss, default is 0.75.
        @type gamma: float
        @param gamma: gamma parameter in focal loss, default is 2.0.
        """

        super().__init__()

        self.alpha = alpha
        self.gamma = gamma

    def forward(
        self, pred_score: Tensor, target_score: Tensor, label: Tensor
    ) -> Tensor:
        weight = (
            self.alpha * pred_score.pow(self.gamma) * (1 - label)
            + target_score * label
        )
        with torch.amp.autocast(
            device_type=pred_score.device.type, enabled=False
        ):
            ce_loss = F.binary_cross_entropy(
                pred_score.float(), target_score.float(), reduction="none"
            )
        loss = (ce_loss * weight).sum()
        return loss
