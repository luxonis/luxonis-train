import torch
import torch.nn.functional as F
from torch import Tensor
from torchvision.ops import box_convert

from luxonis_train.nodes import PrecisionSegmentBBoxHead
from luxonis_train.tasks import Tasks
from luxonis_train.utils import apply_bounding_box_to_masks

from .precision_dlf_segmentation_loss import PrecisionDFLSegmentationLoss


class WeightedPrecisionDFLSegmentationLoss(PrecisionDFLSegmentationLoss):
    node: PrecisionSegmentBBoxHead
    supported_tasks = [Tasks.INSTANCE_SEGMENTATION]

    def __init__(
        self,
        seg_loss_weight: float | None = None,
        mask_pos_weight: float = 1.0,
        **kwargs,
    ):
        """Instance Segmentation loss with configurable mask weighting.

        Extends L{PrecisionDFLSegmentationLoss} with two additional
        parameters for better control over mask loss, especially
        useful for objects where the mask occupies a small fraction of
        the bounding box

        @type seg_loss_weight: float | None
        @param seg_loss_weight: Independent weight for the segmentation
            mask loss. Defaults to C{None}, which falls back to
            C{bbox_loss_weight} (same behavior as PrecisionDFLSegmentationLoss).
        @type mask_pos_weight: float
        @param mask_pos_weight: Weight applied to positive (foreground)
            pixels in the mask BCE loss. Values > 1 increase the
            penalty for missed foreground pixels. For example with mask_pos_weight=9,
            the foreground pixels now contribute 9x more loss each.
            Defaults to 1.0 (no reweighting).
        """
        super().__init__(**kwargs)
        self.seg_loss_weight = (
            seg_loss_weight
            if seg_loss_weight is not None
            else self.bbox_loss_weight
        )
        self.mask_pos_weight = mask_pos_weight

    def forward(
        self,
        features: list[Tensor],
        prototypes: Tensor,
        mask_coeficients: Tensor,
        target_boundingbox: Tensor,
        target_instance_segmentation: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        loss_cls, loss_iou, loss_dfl, loss_seg, batch_size = (
            self._compute_individual_losses(
                features,
                prototypes,
                mask_coeficients,
                target_boundingbox,
                target_instance_segmentation,
            )
        )

        loss = (
            self.class_loss_weight * loss_cls
            + self.bbox_loss_weight * loss_iou
            + self.dfl_loss_weight * loss_dfl
            + self.seg_loss_weight * loss_seg
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
        """Compute the segmentation loss with foreground pixel
        reweighting.

        Same as the PrecisionDFLSegmentationLoss, but applies L{mask_pos_weight} to
        the BCE loss to weight foreground pixels more.
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

        pos_weight = torch.tensor(
            self.mask_pos_weight, device=proto.device, dtype=proto.dtype
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

                pred_mask = torch.einsum("in,nhw->ihw", pred[fg], pr)
                loss = F.binary_cross_entropy_with_logits(
                    pred_mask, gt_mask, reduction="none",
                    pos_weight=pos_weight,
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
