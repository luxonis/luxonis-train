from typing import Annotated

import torch
from pydantic import Field
from torch import Tensor

from luxonis_train.utils.boxutils import process_keypoints_predictions
from luxonis_train.attached_modules.metrics.object_keypoint_similarity import compute_oks
from luxonis_train.utils.types import (
    BaseProtocol,
    Labels,
    LabelType,
    Packet,
)

from .base_loss import BaseLoss
from .bce_with_logits import BCEWithLogitsLoss


class Protocol(BaseProtocol):
    keypoints: Annotated[list[Tensor], Field(min_length=1, max_length=1)]


class KeypointLoss(BaseLoss[Tensor, Tensor]):
    def __init__(
        self,
        bce_power: float = 1.0,
        sigmas: Tensor | None = None,
        **kwargs,
    ):
        super().__init__(
            protocol=Protocol, required_labels=[LabelType.KEYPOINT], **kwargs
        )
        self.b_cross_entropy = BCEWithLogitsLoss(
            pos_weight=torch.tensor([bce_power]), **kwargs
        )
        self.sigmas = sigmas

    def prepare(self, inputs: Packet[Tensor], labels: Labels) -> tuple[Tensor, Tensor]:
        return torch.cat(inputs["keypoints"], dim=0), labels[LabelType.KEYPOINT]

    def forward(
        self, prediction: Tensor, target: Tensor, area: Tensor
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Computes the keypoint loss and visibility loss for a given prediction and
        target.

        @type prediction: Tensor
        @param prediction: Predicted tensor of shape C{[n_detections, n_keypoints * 3]}.
        @type target: Tensor
        @param target: Target tensor of shape C{[n_detections, n_keypoints * 2]}.
        @rtype: tuple[Tensor, Tensor]
        @return: A tuple containing the regression loss tensor of shape C{[1,]} and the
            visibility loss tensor of shape C{[1,]}.
        """
        pred_x, pred_y, pred_v = process_keypoints_predictions(prediction)
        gt_x = target[:, 0::3]
        gt_y = target[:, 1::3]
        gt_v = target[:, 2::3]
        pred = torch.stack((pred_x, pred_y, pred_v), dim=2)
        reshaped_targets = torch.stack((gt_x, gt_y, gt_v), dim=-1)

        mask = target[:, 0::3] != 0
        print(f"mask = {mask}")
        print(f"gt_v = {gt_v}")
        visibility_loss = self.b_cross_entropy.forward(pred_v, mask.float())

        if len(self.sigmas) == 17:
            use_cocoeval = (self.sigmas == torch.tensor(
                [0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 
                0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089], dtype=torch.float32
            )).all()
        else:
            use_cocoeval = False


        okc = compute_oks(pred, reshaped_targets, area * 0.53, use_cocoeval, self.sigmas)
        
        regression_loss_unreduced = 1 - torch.exp(-okc)
        regression_loss_reduced = (
            (regression_loss_unreduced * mask).sum(dim=1, keepdim=False) /
            (mask.sum(dim=1, keepdim=False) + 1e-9)
        )
        regression_loss = regression_loss_reduced.mean()
        loss = regression_loss + visibility_loss
        return loss, {"regression": regression_loss, "visibility": visibility_loss}
