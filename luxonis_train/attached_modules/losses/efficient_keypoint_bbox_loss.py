from typing import Literal, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.ops import box_convert

from luxonis_train.attached_modules.metrics.object_keypoint_similarity import (
    get_area_factor,
    get_sigmas,
)
from luxonis_train.nodes import EfficientKeypointBBoxHead
from luxonis_train.utils.assigners import ATSSAssigner, TaskAlignedAssigner
from luxonis_train.utils.boxutils import (
    IoUType,
    anchors_for_fpn_features,
    compute_iou_loss,
    dist2bbox,
)
from luxonis_train.utils.types import IncompatibleException, Labels, LabelType, Packet

from .base_loss import BaseLoss
from .bce_with_logits import BCEWithLogitsLoss


class EfficientKeypointBBoxLoss(
    BaseLoss[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]
):
    node: EfficientKeypointBBoxHead
    supported_labels = [(LabelType.BOUNDINGBOX, LabelType.KEYPOINTS)]

    class NodePacket(Packet[Tensor]):
        features: list[Tensor]
        class_scores: Tensor
        distributions: Tensor

    def __init__(
        self,
        n_warmup_epochs: int = 4,
        iou_type: IoUType = "giou",
        reduction: Literal["sum", "mean"] = "mean",
        class_bbox_loss_weight: float = 1.0,
        iou_loss_weight: float = 2.5,
        viz_pw: float = 1.0,
        regr_kpts_loss_weight: float = 1.5,
        vis_kpts_loss_weight: float = 1.0,
        sigmas: list[float] | None = None,
        area_factor: float | None = None,
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
        @type class_bbox_loss_weight: float
        @param class_bbox_loss_weight: Weight of classification loss for bounding boxes.
        @type regr_kpts_loss_weight: float
        @param regr_kpts_loss_weight: Weight of regression loss for keypoints.
        @type vis_kpts_loss_weight: float
        @param vis_kpts_loss_weight: Weight of visibility loss for keypoints.
        @type iou_loss_weight: float
        @param iou_loss_weight: Weight of IoU loss.
        @type sigmas: list[float] | None
        @param sigmas: Sigmas used in KeypointLoss for OKS metric. If None then use COCO ones if possible or default ones. Defaults to C{None}.
        @type area_factor: float | None
        @param area_factor: Factor by which we multiply bbox area which is used in KeypointLoss. If None then use default one. Defaults to C{None}.
        @type kwargs: dict
        @param kwargs: Additional arguments to pass to L{BaseLoss}.
        """
        super().__init__(**kwargs)

        if not isinstance(self.node, EfficientKeypointBBoxHead):
            raise IncompatibleException(
                f"Loss `{self.name}` is only "
                "compatible with nodes of type `EfficientKeypointBBoxHead`."
            )
        self.iou_type: IoUType = iou_type
        self.reduction = reduction
        self.n_classes = self.node.n_classes
        self.stride = self.node.stride
        self.grid_cell_size = self.node.grid_cell_size
        self.grid_cell_offset = self.node.grid_cell_offset
        self.original_img_size = self.node.original_in_shape[1:]
        self.n_heads = self.node.n_heads
        self.n_kps = self.node.n_keypoints

        self.b_cross_entropy = BCEWithLogitsLoss(pos_weight=torch.tensor([viz_pw]))
        self.sigmas = get_sigmas(
            sigmas=sigmas, n_keypoints=self.n_kps, class_name=self.name
        )
        self.area_factor = get_area_factor(area_factor, class_name=self.name)

        self.n_warmup_epochs = n_warmup_epochs
        self.atts_assigner = ATSSAssigner(topk=9, n_classes=self.n_classes)
        self.tal_assigner = TaskAlignedAssigner(
            topk=13, n_classes=self.n_classes, alpha=1.0, beta=6.0
        )

        self.varifocal_loss = VarifocalLoss()
        self.class_bbox_loss_weight = class_bbox_loss_weight
        self.iou_loss_weight = iou_loss_weight
        self.regr_kpts_loss_weight = regr_kpts_loss_weight
        self.vis_kpts_loss_weight = vis_kpts_loss_weight

    def prepare(
        self, outputs: Packet[Tensor], labels: Labels
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        feats = self.get_input_tensors(outputs, "features")
        pred_scores = self.get_input_tensors(outputs, "class_scores")[0]
        pred_distri = self.get_input_tensors(outputs, "distributions")[0]
        pred_kpts = self.get_input_tensors(outputs, "keypoints_raw")[0]

        batch_size = pred_scores.shape[0]
        device = pred_scores.device

        target_kpts = self.get_label(labels, LabelType.KEYPOINTS)[0]
        target_bbox = self.get_label(labels, LabelType.BOUNDINGBOX)[0]
        n_kpts = (target_kpts.shape[1] - 2) // 3

        gt_bboxes_scale = torch.tensor(
            [
                self.original_img_size[1],
                self.original_img_size[0],
                self.original_img_size[1],
                self.original_img_size[0],
            ],
            device=device,
        )
        gt_kpts_scale = torch.tensor(
            [
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
        pred_kpts = self.dist2kpts_noscale(
            anchor_points_strided, pred_kpts.view(batch_size, -1, n_kpts, 3)
        )

        target_bbox = self._preprocess_bbox_target(
            target_bbox, batch_size, gt_bboxes_scale
        )

        gt_bbox_labels = target_bbox[:, :, :1]
        gt_xyxy = target_bbox[:, :, 1:]
        mask_gt = (gt_xyxy.sum(-1, keepdim=True) > 0).float()

        if self._epoch < self.n_warmup_epochs:
            (
                assigned_labels,
                assigned_bboxes,
                assigned_scores,
                mask_positive,
                assigned_gt_idx,
            ) = self.atts_assigner(
                anchors,
                n_anchors_list,
                gt_bbox_labels,
                gt_xyxy,
                mask_gt,
                pred_bboxes.detach() * stride_tensor,
            )
        else:
            (
                assigned_labels,
                assigned_bboxes,
                assigned_scores,
                mask_positive,
                assigned_gt_idx,
            ) = self.tal_assigner(
                pred_scores.detach(),
                pred_bboxes.detach() * stride_tensor,
                anchor_points,
                gt_bbox_labels,
                gt_xyxy,
                mask_gt,
            )

        batched_kpts = self._preprocess_kpts_target(
            target_kpts, batch_size, gt_kpts_scale
        )
        assigned_gt_idx_expanded = assigned_gt_idx.unsqueeze(-1).unsqueeze(-1)
        selected_keypoints = batched_kpts.gather(
            1, assigned_gt_idx_expanded.expand(-1, -1, self.n_kps, 3)
        )
        xy_components = selected_keypoints[:, :, :, :2]
        normalized_xy = xy_components / stride_tensor.view(1, -1, 1, 1)
        selected_keypoints = torch.cat(
            (normalized_xy, selected_keypoints[:, :, :, 2:]), dim=-1
        )
        gt_kpt = selected_keypoints[mask_positive]
        pred_kpts = pred_kpts[mask_positive]
        assigned_bboxes = assigned_bboxes / stride_tensor

        area = (
            assigned_bboxes[mask_positive][:, 0] - assigned_bboxes[mask_positive][:, 2]
        ) * (
            assigned_bboxes[mask_positive][:, 1] - assigned_bboxes[mask_positive][:, 3]
        )

        return (
            pred_bboxes,
            pred_scores,
            assigned_bboxes,
            assigned_labels,
            assigned_scores,
            mask_positive,
            gt_kpt,
            pred_kpts,
            area * self.area_factor,
        )

    def forward(
        self,
        pred_bboxes: Tensor,
        pred_scores: Tensor,
        assigned_bboxes: Tensor,
        assigned_labels: Tensor,
        assigned_scores: Tensor,
        mask_positive: Tensor,
        gt_kpts: Tensor,
        pred_kpts: Tensor,
        area: Tensor,
    ):
        device = pred_bboxes.device
        sigmas = self.sigmas.to(device)
        d = (gt_kpts[..., 0] - pred_kpts[..., 0]).pow(2) + (
            gt_kpts[..., 1] - pred_kpts[..., 1]
        ).pow(2)
        e = d / ((2 * sigmas).pow(2) * ((area.view(-1, 1) + 1e-9) * 2))
        mask = (gt_kpts[..., 2] > 0).float()
        regression_loss = (
            ((1 - torch.exp(-e)) * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-9)
        ).mean()
        visibility_loss = self.b_cross_entropy.forward(pred_kpts[..., 2], mask)

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

        loss = (
            self.class_bbox_loss_weight * loss_cls
            + self.iou_loss_weight * loss_iou
            + regression_loss * self.regr_kpts_loss_weight
            + visibility_loss * self.vis_kpts_loss_weight
        )

        sub_losses = {
            "class": loss_cls.detach(),
            "iou": loss_iou.detach(),
            "regression": regression_loss.detach(),
            "visibility": visibility_loss.detach(),
        }

        return loss, sub_losses

    def _preprocess_bbox_target(
        self, bbox_target: Tensor, batch_size: int, scale_tensor: Tensor
    ) -> Tensor:
        """Preprocess target bboxes in shape [batch_size, N, 5] where N is maximum
        number of instances in one image."""
        sample_ids, counts = cast(
            tuple[Tensor, Tensor],
            torch.unique(bbox_target[:, 0].int(), return_counts=True),
        )
        c_max = int(counts.max()) if counts.numel() > 0 else 0
        out_target = torch.zeros(batch_size, c_max, 5, device=bbox_target.device)
        out_target[:, :, 0] = -1
        for id, count in zip(sample_ids, counts):
            out_target[id, :count] = bbox_target[bbox_target[:, 0] == id][:, 1:]

        scaled_target = out_target[:, :, 1:5] * scale_tensor
        out_target[..., 1:] = box_convert(scaled_target, "xywh", "xyxy")
        return out_target

    def _preprocess_kpts_target(
        self, kpts_target: Tensor, batch_size: int, scale_tensor: Tensor
    ) -> Tensor:
        """Preprocesses the target keypoints in shape [batch_size, N, n_keypoints, 3]
        where N is the maximum number of keypoints in one image."""

        _, counts = torch.unique(kpts_target[:, 0].int(), return_counts=True)
        max_kpts = int(counts.max()) if counts.numel() > 0 else 0
        batched_keypoints = torch.zeros(
            (batch_size, max_kpts, self.n_kps, 3), device=kpts_target.device
        )
        for i in range(batch_size):
            keypoints_i = kpts_target[kpts_target[:, 0] == i]
            scaled_keypoints_i = keypoints_i[:, 2:].clone()
            batched_keypoints[i, : keypoints_i.shape[0]] = scaled_keypoints_i.view(
                -1, self.n_kps, 3
            )
            batched_keypoints[i, :, :, :2] *= scale_tensor[:2]

        return batched_keypoints

    def dist2kpts_noscale(self, anchor_points: Tensor, kpts: Tensor) -> Tensor:
        """Adjusts and scales predicted keypoints relative to anchor points without
        considering image stride."""
        adj_kpts = kpts.clone()
        scale = 2.0
        x_adj = anchor_points[:, [0]] - 0.5
        y_adj = anchor_points[:, [1]] - 0.5

        adj_kpts[..., :2] *= scale
        adj_kpts[..., 0] += x_adj
        adj_kpts[..., 1] += y_adj
        return adj_kpts


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
