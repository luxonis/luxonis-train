from typing import Literal, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from luxonis_train.nodes.heads import EfficientOBBoxHead
from luxonis_train.utils.assigners import RotatedTaskAlignedAssigner
from luxonis_train.utils.boxutils import (
    IoUType,
    anchors_for_fpn_features,
    bbox2dist,
    dist2rbbox,
    probiou,
    xywh2xyxy,
    xyxyxyxy2xywhr,
)
from luxonis_train.utils.types import IncompatibleException, Labels, LabelType, Packet

from .base_loss import BaseLoss


class OBBDetectionLoss(BaseLoss[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]):
    node: EfficientOBBoxHead
    supported_labels = [LabelType.OBOUNDINGBOX]

    class NodePacket(Packet[Tensor]):
        features: list[Tensor]
        class_scores: Tensor
        distributions: Tensor
        angles: Tensor

    def __init__(
        self,
        iou_type: IoUType = "giou",
        reduction: Literal["sum", "mean"] = "mean",
        class_loss_weight: float = 1.0,
        iou_loss_weight: float = 2.5,
        dfl_loss_weight: float = 1.0,
        reg_max: int = 16,
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
        super().__init__(**kwargs)

        if not isinstance(self.node, EfficientOBBoxHead):
            raise IncompatibleException(
                f"Loss `{self.name}` is only "
                "compatible with nodes of type `EfficientOBBoxHead`."
            )
        self.iou_type: IoUType = iou_type
        self.reduction = reduction
        self.n_classes = self.node.n_classes
        self.stride = self.node.stride
        self.grid_cell_size = self.node.grid_cell_size
        self.grid_cell_offset = self.node.grid_cell_offset
        self.original_img_size = self.node.original_in_shape[1:]
        self.reg_max = reg_max

        self.assigner = RotatedTaskAlignedAssigner(
            n_classes=self.n_classes, topk=10, alpha=0.5, beta=6.0
        )
        # Bounding box loss
        self.bbox_loss = RotatedBboxLoss(self.reg_max)
        # Class loss
        self.varifocal_loss = VarifocalLoss()
        # self.bce = nn.BCEWithLogitsLoss(reduction="none")

        # self.n_warmup_epochs = n_warmup_epochs
        # self.atts_assigner = ATSSAssigner(topk=9, n_classes=self.n_classes)
        # self.tal_assigner = TaskAlignedAssigner(
        #     topk=13, n_classes=self.n_classes, alpha=1.0, beta=6.0
        # )

        self.class_loss_weight = class_loss_weight
        self.iou_loss_weight = iou_loss_weight
        self.dfl_loss_weight = dfl_loss_weight

        self.anchors = None
        self.anchor_points = None
        self.n_anchors_list = None
        self.stride_tensor = None
        self.gt_bboxes_scale = None

    def prepare(
        self, outputs: Packet[Tensor], labels: Labels
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        feats = self.get_input_tensors(outputs, "features")
        pred_scores = self.get_input_tensors(outputs, "class_scores")[0]
        self.pred_distri = self.get_input_tensors(outputs, "distributions")[0]
        pred_angles = self.get_input_tensors(outputs, "angles")[0]
        batch_size = pred_scores.shape[0]
        device = pred_scores.device

        target = self.get_label(labels)[0]
        if self.gt_bboxes_scale is None:
            self.gt_bboxes_scale = torch.tensor(
                [
                    self.original_img_size[1],
                    self.original_img_size[0],
                    self.original_img_size[1],
                    self.original_img_size[0],
                ],
                device=device,
            )
            (
                self.anchors,
                self.anchor_points,
                self.n_anchors_list,
                self.stride_tensor,
            ) = anchors_for_fpn_features(
                feats,
                self.stride,
                self.grid_cell_size,
                self.grid_cell_offset,
                multiply_with_stride=True,
            )
            self.anchor_points_strided = self.anchor_points / self.stride_tensor

        target = self._preprocess_target(
            target, batch_size
        )  # [cls, x, y, w, h, r] unnormalized

        proj = torch.arange(
            self.reg_max, dtype=torch.float, device=self.pred_distri.device
        )
        b, a, c = self.pred_distri.shape  # batch, anchors, channels
        pred_distri_tensor = (  # we get a tensor of the expected values (mean) of the regression predictions
            self.pred_distri.view(b, a, 4, c // 4)
            .softmax(3)
            .matmul(proj.type(self.pred_distri.dtype))
        )
        pred_bboxes = torch.cat(
            (
                dist2rbbox(pred_distri_tensor, pred_angles, self.anchor_points_strided),
                pred_angles,
            ),
            dim=-1,
        )  # xywhr unnormalized

        xy_strided = pred_bboxes[..., :2] * self.stride_tensor
        pred_bboxes_strided = torch.cat(
            [xy_strided, pred_bboxes[..., 2:]], dim=-1
        )  # xywhr unnormalized with xy strided

        gt_cls = target[:, :, :1]
        gt_cxcywhr = target[:, :, 1:]
        mask_gt = (gt_cxcywhr.sum(-1, keepdim=True) > 0).float()

        # TODO: log change of assigner (once common Logger)
        (
            assigned_labels,
            assigned_bboxes,
            assigned_scores,
            mask_positive,
            _,
        ) = self.assigner(
            pred_scores.detach(),
            pred_bboxes_strided.detach(),
            self.anchor_points,
            gt_cls,
            gt_cxcywhr,
            mask_gt,
        )

        xy_unstrided = assigned_bboxes[..., :2] / self.stride_tensor
        assigned_bboxes_unstrided = torch.cat(
            [xy_unstrided, assigned_bboxes[..., 2:]], dim=-1
        )  # xywhr unnormalized with xy strided

        return (
            pred_bboxes,
            pred_scores,
            assigned_bboxes_unstrided,
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

        # CLS loss
        loss_cls = self.varifocal_loss(pred_scores, assigned_scores, one_hot_label)
        if assigned_scores.sum() > 1:
            loss_cls /= assigned_scores.sum()

        assigned_scores_sum = max(assigned_scores.sum(), 1)
        # Bbox loss
        self.bbox_loss = self.bbox_loss.to(self.pred_distri.device)
        loss_iou, loss_dfl = self.bbox_loss(
            self.pred_distri,
            pred_bboxes,
            self.anchor_points,
            assigned_bboxes,
            assigned_scores,
            assigned_scores_sum,
            mask_positive,
        )

        loss = (
            self.class_loss_weight * loss_cls
            + self.iou_loss_weight * loss_iou
            + self.dfl_loss_weight * loss_dfl
        )

        sub_losses = {
            "class": loss_cls.detach(),
            "iou": loss_iou.detach(),
            "dfl": loss_dfl.detach(),
        }

        return loss, sub_losses

    def _preprocess_target(self, target: Tensor, batch_size: int):
        """Preprocess target in shape [batch_size, N, 6] where N is maximum number of
        instances in one image."""
        idx_cls = target[:, :2]
        xyxyxyxy = target[:, 2:]
        cxcywhr = xyxyxyxy2xywhr(xyxyxyxy)
        if isinstance(cxcywhr, Tensor):
            target = torch.cat([idx_cls, cxcywhr.clone().detach()], dim=-1)
        else:
            target = torch.cat([idx_cls, torch.tensor(cxcywhr)], dim=-1)
        sample_ids, counts = cast(
            tuple[Tensor, Tensor], torch.unique(target[:, 0].int(), return_counts=True)
        )
        c_max = int(counts.max()) if counts.numel() > 0 else 0
        out_target = torch.zeros(batch_size, c_max, 6, device=target.device)
        out_target[:, :, 0] = -1
        for id, count in zip(sample_ids, counts):
            out_target[id, :count] = target[target[:, 0] == id][:, 1:]

        scaled_target = out_target[:, :, 1:5] * self.gt_bboxes_scale
        scaled_target_angle = torch.cat(
            [scaled_target, out_target[:, :, 5].transpose(0, 1).unsqueeze(0)], dim=-1
        )
        # out_target[..., 1:] = box_convert(scaled_target, "xywh", "xyxy")
        out_target[..., 1:] = scaled_target_angle
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


class DFLoss(nn.Module):
    """Criterion class for computing DFL losses during training."""

    def __init__(self, reg_max=16) -> None:
        """Initialize the DFL module."""
        super().__init__()
        self.reg_max = reg_max

    def __call__(self, pred_dist, target):
        """Return sum of left and right DFL losses.

        Distribution Focal Loss (DFL) proposed in Generalized Focal Loss
        https://ieeexplore.ieee.org/document/9792391
        """
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl = target.long()  # target left
        # tl = target  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape)
            * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape)
            * wr
        ).mean(-1, keepdim=True)


class RotatedBboxLoss(nn.Module):
    """Criterion class for computing training losses during training."""

    def __init__(self, reg_max):
        """Initialize the BboxLoss module with regularization maximum and DFL
        settings."""
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None

    def forward(
        self,
        pred_dist,
        pred_bboxes,
        anchor_points,
        target_bboxes,
        target_scores,
        target_scores_sum,
        fg_mask,
    ):
        """IoU loss."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = probiou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(
                xywh2xyxy(target_bboxes[..., :4]),
                anchor_points,
                self.dfl_loss.reg_max - 1,
            )
            loss_dfl = (
                self.dfl_loss(
                    pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max),
                    target_ltrb[fg_mask],
                )
                * weight
            )
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl
