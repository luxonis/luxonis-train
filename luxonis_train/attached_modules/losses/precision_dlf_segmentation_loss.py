import torch
import torch.nn.functional as F
from torch import Tensor
from torchvision.ops import box_convert

from luxonis_train.nodes import PrecisionSegmentBBoxHead
from luxonis_train.tasks import Tasks
from luxonis_train.utils import apply_bounding_box_to_masks

from .precision_dfl_detection_loss import PrecisionDFLDetectionLoss


class PrecisionDFLSegmentationLoss(PrecisionDFLDetectionLoss):
    node: PrecisionSegmentBBoxHead
    supported_tasks = [Tasks.INSTANCE_SEGMENTATION]

    def __init__(
        self,
        tal_topk: int = 10,
        class_loss_weight: float = 0.5,
        bbox_loss_weight: float = 7.5,
        dfl_loss_weight: float = 1.5,
        **kwargs,
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

    def forward(
        self,
        features: list[Tensor],
        prototypes: Tensor,
        mask_coeficients: Tensor,
        target_boundingbox: Tensor,
        target_instance_segmentation: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
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
        mask_coeficients = mask_coeficients.permute(0, 2, 1).contiguous()

        target_boundingbox = self._preprocess_bbox_target(
            target_boundingbox, batch_size
        )

        pred_bboxes = self.decode_bbox(self.anchor_points_strided, pred_distri)

        gt_labels = target_boundingbox[:, :, :1]
        gt_xyxy = target_boundingbox[:, :, 1:]
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
            target_instance_segmentation,
            assigned_gt_idx,
            assigned_bboxes,
            img_idx,
            prototypes,
            mask_coeficients,
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

        return loss * batch_size, sub_losses

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
        """Compute the segmentation loss for the entire batch.

        @type fg_mask: Tensor
        @param fg_mask: Foreground mask. Shape: (B, N_anchor).
        @type gt_masks: Tensor
        @param gt_masks: Ground truth masks. Shape: (n, H, W).
        @type gt_idx: Tensor
        @param gt_idx: Ground truth mask indices. Shape: (B, N_anchor).
        @type bboxes: Tensor
        @param bboxes: Ground truth bounding boxes in xyxy format.
            Shape: (B, N_anchor, 4).
        @type batch_ids: Tensor
        @param batch_ids: Batch indices. Shape: (n, 1).
        @type proto: Tensor
        @param proto: Prototype masks. Shape: (B, 32, H, W).
        @type pred_masks: Tensor
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
