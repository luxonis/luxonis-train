from typing import Literal

import torch
import torch.nn.functional as F
from loguru import logger
from torch import Tensor

from luxonis_train.attached_modules.losses import AdaptiveDetectionLoss
from luxonis_train.nodes import EfficientKeypointBBoxHead
from luxonis_train.tasks import Tasks
from luxonis_train.utils import (
    compute_iou_loss,
    dist2bbox,
    get_sigmas,
    get_with_default,
)
from luxonis_train.utils.boundingbox import IoUType
from luxonis_train.utils.keypoints import insert_class

from .bce_with_logits import BCEWithLogitsLoss


class EfficientKeypointBBoxLoss(AdaptiveDetectionLoss):
    node: EfficientKeypointBBoxHead
    supported_tasks = [Tasks.INSTANCE_KEYPOINTS]

    gt_kpts_scale: Tensor

    def __init__(
        self,
        n_warmup_epochs: int = 0,
        iou_type: IoUType = "giou",
        reduction: Literal["sum", "mean"] = "mean",
        class_loss_weight: float = 0.5,
        iou_loss_weight: float = 7.5,
        viz_pw: float = 1.0,
        regr_kpts_loss_weight: float = 12,
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
        @type iou_type: Literal["none", "giou", "diou", "ciou", "siou"]
        @param iou_type: IoU type used for bbox regression loss.
        @type reduction: Literal["sum", "mean"]
        @param reduction: Reduction type for loss.
        @type class_loss_weight: float
        @param class_loss_weight: Weight of classification loss for bounding boxes.
        @type regr_kpts_loss_weight: float
        @param regr_kpts_loss_weight: Weight of regression loss for keypoints. Defaults to 12.0. For optimal results, multiply with accumulate_grad_batches.
        @type vis_kpts_loss_weight: float
        @param vis_kpts_loss_weight: Weight of visibility loss for keypoints. Defaults to 1.0. For optimal results, multiply with accumulate_grad_batches.
        @type iou_loss_weight: float
        @param iou_loss_weight: Weight of IoU loss. Defaults to 2.5. For optimal results, multiply with accumulate_grad_batches.
        @type sigmas: list[float] | None
        @param sigmas: Sigmas used in keypoint loss for OKS metric. If None then use COCO ones if possible or default ones. Defaults to C{None}.
        @type area_factor: float | None
        @param area_factor: Factor by which we multiply bounding box area which is used in the keypoint loss.
            If not set, the default factor of `0.53` is used.
        """
        super().__init__(
            n_warmup_epochs=n_warmup_epochs,
            iou_type=iou_type,
            reduction=reduction,
            class_loss_weight=class_loss_weight,
            iou_loss_weight=iou_loss_weight,
            **kwargs,
        )

        self.b_cross_entropy = BCEWithLogitsLoss(
            pos_weight=torch.tensor([viz_pw])
        )
        self.sigmas = get_sigmas(
            sigmas=sigmas, n_keypoints=self.n_keypoints, caller_name=self.name
        )
        self.area_factor = get_with_default(
            area_factor, "bbox area scaling", self.name, default=0.53
        )
        self.regr_kpts_loss_weight = regr_kpts_loss_weight
        self.vis_kpts_loss_weight = vis_kpts_loss_weight

    def forward(
        self,
        features: list[Tensor],
        class_scores: Tensor,
        distributions: Tensor,
        keypoints_raw: Tensor,
        target_boundingbox: Tensor,
        target_keypoints: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        device = keypoints_raw.device
        target_keypoints = insert_class(target_keypoints, target_boundingbox)

        batch_size = class_scores.shape[0]
        n_kpts = (target_keypoints.shape[1] - 2) // 3

        self._init_parameters(features)

        pred_bboxes = dist2bbox(distributions, self.anchor_points_strided)
        keypoints_raw = self.dist2kpts_noscale(
            self.anchor_points_strided,
            keypoints_raw.view(batch_size, -1, n_kpts, 3),
        )

        target_boundingbox = self._preprocess_bbox_target(
            target_boundingbox, batch_size
        )

        gt_bbox_labels = target_boundingbox[..., :1]
        gt_xyxy = target_boundingbox[..., 1:]
        mask_gt = (gt_xyxy.sum(-1, keepdim=True) > 0).float()

        batched_kpts = self._preprocess_kpts_target(
            target_keypoints, batch_size, self.gt_kpts_scale
        )

        scaled_raw_keypoints = keypoints_raw.clone()
        scaled_raw_keypoints[..., :2] = scaled_raw_keypoints[
            ..., :2
        ] * self.stride_tensor.view(1, -1, 1, 1)

        sigmas = self.sigmas.to(device)

        (
            assigned_labels,
            assigned_bboxes,
            assigned_scores,
            mask_positive,
            assigned_gt_idx,
        ) = self._run_assigner(
            gt_bbox_labels,
            gt_xyxy,
            mask_gt,
            pred_bboxes,
            class_scores,
            scaled_raw_keypoints,
            batched_kpts,
            sigmas,
            self.area_factor,
        )

        assigned_gt_idx_expanded = assigned_gt_idx.unsqueeze(-1).unsqueeze(-1)
        if batched_kpts.numel() == 0:
            logger.debug("No instances found in the batch")
            selected_keypoints = batched_kpts.new_zeros(
                (
                    batched_kpts.size(0),
                    assigned_gt_idx_expanded.size(1),
                    self.n_keypoints,
                    3,
                )
            )
        else:
            selected_keypoints = batched_kpts.gather(
                1, assigned_gt_idx_expanded.expand(-1, -1, self.n_keypoints, 3)
            )
        xy_components = selected_keypoints[..., :2]
        normalized_xy = xy_components / self.stride_tensor.view(1, -1, 1, 1)
        selected_keypoints = torch.cat(
            (normalized_xy, selected_keypoints[..., 2:]), dim=-1
        )
        gt_kpts = selected_keypoints[mask_positive]
        keypoints_raw = keypoints_raw[mask_positive]
        assigned_bboxes = assigned_bboxes / self.stride_tensor

        area = (
            (
                assigned_bboxes[mask_positive][:, 0]
                - assigned_bboxes[mask_positive][:, 2]
            )
            * (
                assigned_bboxes[mask_positive][:, 1]
                - assigned_bboxes[mask_positive][:, 3]
            )
            * self.area_factor
        )

        d = (gt_kpts[..., 0] - keypoints_raw[..., 0]).pow(2) + (
            gt_kpts[..., 1] - keypoints_raw[..., 1]
        ).pow(2)
        e = d / ((2 * sigmas).pow(2) * ((area.view(-1, 1) + 1e-9) * 2))
        mask = (gt_kpts[..., 2] > 0).float()
        regression_loss = (
            ((1 - torch.exp(-e)) * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-9)
        ).mean()
        visibility_loss = self.b_cross_entropy.forward(
            keypoints_raw[..., 2], mask
        )

        one_hot_label = F.one_hot(assigned_labels.long(), self.n_classes + 1)[
            ..., :-1
        ]
        loss_cls = self.varifocal_loss(
            class_scores, assigned_scores, one_hot_label
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
            self.class_loss_weight * loss_cls
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

        return loss * batch_size, sub_losses

    def _preprocess_kpts_target(
        self, kpts_target: Tensor, batch_size: int, scale_tensor: Tensor
    ) -> Tensor:
        """Preprocesses the target keypoints in shape [batch_size, N,
        n_keypoints, 3] where N is the maximum number of keypoints in
        one image."""
        _, counts = torch.unique(kpts_target[:, 0].int(), return_counts=True)
        max_kpts = int(counts.max()) if counts.numel() > 0 else 0
        batched_keypoints = torch.zeros(
            (batch_size, max_kpts, self.n_keypoints, 3),
            device=kpts_target.device,
        )
        for i in range(batch_size):
            keypoints_i = kpts_target[kpts_target[:, 0] == i]
            scaled_keypoints_i = keypoints_i[:, 2:].clone()
            batched_keypoints[i, : keypoints_i.shape[0]] = (
                scaled_keypoints_i.view(-1, self.n_keypoints, 3)
            )
            batched_keypoints[i, :, :, :2] *= scale_tensor[:2]

        return batched_keypoints

    def dist2kpts_noscale(self, anchor_points: Tensor, kpts: Tensor) -> Tensor:
        """Adjusts and scales predicted keypoints relative to anchor
        points without considering image stride."""
        adj_kpts = kpts.clone()
        scale = 2.0
        x_adj = anchor_points[:, [0]] - 0.5
        y_adj = anchor_points[:, [1]] - 0.5

        adj_kpts[..., :2] *= scale
        adj_kpts[..., 0] += x_adj
        adj_kpts[..., 1] += y_adj
        return adj_kpts

    def _init_parameters(self, features: list[Tensor]) -> None:
        if hasattr(self, "gt_kpts_scale"):
            return
        super()._init_parameters(features)
        self.gt_kpts_scale = torch.tensor(
            [self.original_img_size[1], self.original_img_size[0]],
            device=features[0].device,
        )
