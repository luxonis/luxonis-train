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
    node: PrecisionBBoxHead
    supported_tasks = [Tasks.BOUNDINGBOX]

    def __init__(
        self,
        tal_topk: int = 10,
        class_loss_weight: float = 0.5,
        bbox_loss_weight: float = 7.5,
        dfl_loss_weight: float = 1.5,
        **kwargs,
    ):
        """BBox loss adapted from  U{Real-Time Flying Object Detection with YOLOv8
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
        super().__init__(**kwargs)
        self.stride = self.node.stride
        self.grid_cell_size = self.node.grid_cell_size
        self.grid_cell_offset = self.node.grid_cell_offset
        self.original_img_size = self.original_in_shape[1:]

        self.class_loss_weight = class_loss_weight
        self.bbox_loss_weight = bbox_loss_weight
        self.dfl_loss_weight = dfl_loss_weight

        self.assigner = TaskAlignedAssigner(
            n_classes=self.n_classes, topk=tal_topk, alpha=0.5, beta=6.0
        )
        self.bbox_loss = BBoxLoss(self.node.reg_max)
        self.proj = torch.arange(self.node.reg_max, dtype=torch.float)
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(
        self, features: list[Tensor], target: Tensor
    ) -> tuple[Tensor, dict[str, Tensor]]:
        self._init_parameters(features)
        batch_size = features[0].shape[0]
        pred_distri, pred_scores = torch.cat(
            [xi.view(batch_size, self.node.no, -1) for xi in features], 2
        ).split((self.node.reg_max * 4, self.n_classes), 1)
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()

        target = self._preprocess_bbox_target(target, batch_size)

        pred_bboxes = self.decode_bbox(self.anchor_points_strided, pred_distri)

        gt_labels = target[:, :, :1]
        gt_xyxy = target[:, :, 1:]
        mask_gt = (gt_xyxy.sum(-1, keepdim=True) > 0).float()

        _, assigned_bboxes, assigned_scores, mask_positive, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * self.stride_tensor).type(gt_xyxy.dtype),
            self.anchor_points,
            gt_labels,
            gt_xyxy,
            mask_gt,
        )
        assigned_bboxes /= self.stride_tensor

        max_assigned_scores_sum = max(assigned_scores.sum().item(), 1)
        loss_cls = (
            self.bce(pred_scores, assigned_scores)
        ).sum() / max_assigned_scores_sum
        if mask_positive.sum():
            loss_iou, loss_dfl = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                self.anchor_points_strided,
                assigned_bboxes,
                assigned_scores,
                max_assigned_scores_sum,
                mask_positive,
            )
        else:
            loss_iou = torch.tensor(0.0).to(pred_distri.device)
            loss_dfl = torch.tensor(0.0).to(pred_distri.device)

        loss = (
            self.class_loss_weight * loss_cls
            + self.bbox_loss_weight * loss_iou
            + self.dfl_loss_weight * loss_dfl
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
        """Decode predicted object bounding box coordinates from anchor
        points and distribution.

        @type anchor_points: Tensor
        @param anchor_points: Anchor points tensor of shape [N, 4] where
            N is the number of anchors.
        @type pred_dist: Tensor
        @param pred_dist: Predicted distribution tensor of shape
            [batch_size, N, 4 * reg_max] where N is the number of
            anchors.
        @rtype: Tensor
        """
        if self.node.dfl:
            batch_size, n_anchors, n_channels = pred_dist.shape
            dist_probs = pred_dist.view(
                batch_size, n_anchors, 4, n_channels // 4
            ).softmax(dim=3)
            dist_transformed = dist_probs @ self.proj.to(
                anchor_points.device, dtype=pred_dist.dtype
            )
        return dist2bbox(dist_transformed, anchor_points, out_format="xyxy")

    def _init_parameters(self, features: list[Tensor]) -> None:
        if not hasattr(self, "gt_bboxes_scale"):
            _, self.anchor_points, _, self.stride_tensor = (
                anchors_for_fpn_features(
                    features,
                    self.stride,
                    self.grid_cell_size,
                    self.grid_cell_offset,
                    multiply_with_stride=True,
                )
            )
            self.gt_bboxes_scale = torch.tensor(
                [
                    self.original_img_size[1],
                    self.original_img_size[0],
                    self.original_img_size[1],
                    self.original_img_size[0],
                ],
                device=features[0].device,
            )
            self.anchor_points_strided = (
                self.anchor_points / self.stride_tensor
            )


class BBoxLoss(nn.Module):
    def __init__(self, reg_max: int = 16):
        """BBox loss that combines IoU and DFL losses.

        @type reg_max: int
        @param reg_max: Maximum number of regression channels. Defaults
            to 16.
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
    def __init__(self, reg_max: int = 16):
        """DFL loss that combines classification and regression losses.

        @type reg_max: int
        @param reg_max: Maximum number of regression channels. Defaults
            to 16.
        """
        super().__init__()
        self.reg_max = reg_max

    def __call__(self, pred_dist: Tensor, targets: Tensor) -> Tensor:
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
