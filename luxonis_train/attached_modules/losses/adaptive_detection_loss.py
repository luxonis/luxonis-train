from typing import Literal, cast

import torch
import torch.nn.functional as F
from pydantic import Field
from torch import Tensor, nn
from torchvision.ops import box_convert
from typing_extensions import Annotated

from luxonis_train.nodes import EfficientBBoxHead
from luxonis_train.utils.assigners import ATSSAssigner, TaskAlignedAssigner
from luxonis_train.utils.boxutils import (
    IoUType,
    anchors_for_fpn_features,
    compute_iou_loss,
    dist2bbox,
)
from luxonis_train.utils.types import (
    BaseProtocol,
    IncompatibleException,
    Labels,
    LabelType,
    Packet,
)

from .base_loss import BaseLoss


class Protocol(BaseProtocol):
    features: list[Tensor]
    class_scores: Annotated[list[Tensor], Field(min_length=1, max_length=1)]
    distributions: Annotated[list[Tensor], Field(min_length=1, max_length=1)]


class AdaptiveDetectionLoss(BaseLoss[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]):
    node: EfficientBBoxHead

    class NodePacket(Packet[Tensor]):
        features: list[Tensor]
        class_scores: Tensor
        distributions: Tensor

    def __init__(
        self,
        n_warmup_epochs: int = 4,
        iou_type: IoUType = "giou",
        reduction: Literal["sum", "mean"] = "mean",
        class_loss_weight: float = 1.0,
        iou_loss_weight: float = 2.5,
        **kwargs,
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
        @type kwargs: dict
        @param kwargs: Additional arguments to pass to L{BaseLoss}.
        """
        super().__init__(
            required_labels=[LabelType.BOUNDINGBOX], protocol=Protocol, **kwargs
        )

        if not isinstance(self.node, EfficientBBoxHead):
            raise IncompatibleException(
                f"Loss `{self.__class__.__name__}` is only "
                "compatible with nodes of type `EfficientBBoxHead`."
            )
        self.iou_type: IoUType = iou_type
        self.reduction = reduction
        self.n_classes = self.node.n_classes
        self.stride = self.node.stride
        self.grid_cell_size = self.node.grid_cell_size
        self.grid_cell_offset = self.node.grid_cell_offset
        self.original_img_size = self.node.original_in_shape[2:]

        self.n_warmup_epochs = n_warmup_epochs
        self.atts_assigner = ATSSAssigner(topk=9, n_classes=self.n_classes)
        self.tal_assigner = TaskAlignedAssigner(
            topk=13, n_classes=self.n_classes, alpha=1.0, beta=6.0
        )

        self.varifocal_loss = VarifocalLoss()
        self.class_loss_weight = class_loss_weight
        self.iou_loss_weight = iou_loss_weight

    def prepare(
        self, outputs: Packet[Tensor], labels: Labels
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        feats = outputs["features"]
        pred_scores = outputs["class_scores"][0]
        pred_distri = outputs["distributions"][0]

        batch_size = pred_scores.shape[0]
        device = pred_scores.device

        target = labels[LabelType.BOUNDINGBOX].to(device)
        gt_bboxes_scale = torch.tensor(
            [
                self.original_img_size[1],
                self.original_img_size[0],
                self.original_img_size[1],
                self.original_img_size[0],
            ],
            device=device,
        )
        (
            anchors,
            anchor_points,
            n_anchors_list,
            stride_tensor,
        ) = anchors_for_fpn_features(
            feats,
            self.stride,
            self.grid_cell_size,
            self.grid_cell_offset,
            multiply_with_stride=True,
        )

        anchor_points_strided = anchor_points / stride_tensor
        pred_bboxes = dist2bbox(pred_distri, anchor_points_strided)

        target = self._preprocess_target(target, batch_size, gt_bboxes_scale)

        gt_labels = target[:, :, :1]
        gt_xyxy = target[:, :, 1:]
        mask_gt = (gt_xyxy.sum(-1, keepdim=True) > 0).float()

        if self._epoch < self.n_warmup_epochs:
            (
                assigned_labels,
                assigned_bboxes,
                assigned_scores,
                mask_positive,
            ) = self.atts_assigner(
                anchors,
                n_anchors_list,
                gt_labels,
                gt_xyxy,
                mask_gt,
                pred_bboxes.detach() * stride_tensor,
            )
        else:
            # TODO: log change of assigner (once common Logger)
            (
                assigned_labels,
                assigned_bboxes,
                assigned_scores,
                mask_positive,
            ) = self.tal_assigner.forward(
                pred_scores.detach(),
                pred_bboxes.detach() * stride_tensor,
                anchor_points,
                gt_labels,
                gt_xyxy,
                mask_gt,
            )

        return (
            pred_bboxes,
            pred_scores,
            assigned_bboxes / stride_tensor,
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
        one_hot_label = F.one_hot(assigned_labels.long(), self.n_classes + 1)[..., :-1]
        loss_cls = self.varifocal_loss(pred_scores, assigned_scores, one_hot_label)

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

        loss = self.class_loss_weight * loss_cls + self.iou_loss_weight * loss_iou

        sub_losses = {"class": loss_cls.detach(), "iou": loss_iou.detach()}

        return loss, sub_losses

    def _preprocess_target(self, target: Tensor, batch_size: int, scale_tensor: Tensor):
        """Preprocess target in shape [batch_size, N, 5] where N is maximum number of
        instances in one image."""
        sample_ids, counts = cast(
            tuple[Tensor, Tensor], torch.unique(target[:, 0].int(), return_counts=True)
        )
        c_max = int(counts.max()) if counts.numel() > 0 else 0
        out_target = torch.zeros(batch_size, c_max, 5, device=target.device)
        out_target[:, :, 0] = -1
        for id, count in zip(sample_ids, counts):
            out_target[id, :count] = target[target[:, 0] == id][:, 1:]

        scaled_target = out_target[:, :, 1:5] * scale_tensor
        out_target[..., 1:] = box_convert(scaled_target, "xywh", "xyxy")
        return out_target


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
            self.alpha * pred_score.pow(self.gamma) * (1 - label) + target_score * label
        )
        ce_loss = F.binary_cross_entropy(
            pred_score.float(), target_score.float(), reduction="none"
        )
        loss = (ce_loss * weight).sum()
        return loss
