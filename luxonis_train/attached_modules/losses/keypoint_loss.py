from typing import Annotated

import torch
from pydantic import Field
from torch import Tensor

from luxonis_train.utils.boxutils import process_keypoints_predictions
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
        distance_weight: float = 0.1,
        visibility_weight: float = 0.6,
        **kwargs,
    ):
        super().__init__(
            protocol=Protocol, required_labels=[LabelType.KEYPOINT], **kwargs
        )
        self.b_cross_entropy = BCEWithLogitsLoss(
            pos_weight=torch.tensor([bce_power]), **kwargs
        )
        self.distance_weight = distance_weight
        self.visibility_weight = visibility_weight

    def prepare(self, inputs: Packet[Tensor], labels: Labels) -> tuple[Tensor, Tensor]:
        return torch.cat(inputs["keypoints"], dim=0), labels[LabelType.KEYPOINT]

    def forward(
        self, prediction: Tensor, target: Tensor
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Computes the keypoint loss and visibility loss for a given prediction and
        target.

        @type prediction: Tensor
        @param prediction: Predicted tensor of shape C{[n_detections, n_keypoints * 3]}.
        @type target: Tensor
        @param target: Target tensor of shape C{[n_detections, n_keypoints * 2]}.
        @rtype: tuple[Tensor, Tensor]
        @return: A tuple containing the keypoint loss tensor of shape C{[1,]} and the
            visibility loss tensor of shape C{[1,]}.
        """
        x, y, visibility_score = process_keypoints_predictions(prediction)
        gt_x = target[:, 0::2]
        gt_y = target[:, 1::2]

        mask = target[:, 0::2] != 0
        visibility_loss = (
            self.b_cross_entropy.forward(visibility_score, mask.float())
            * self.visibility_weight
        )
        distance = (x - gt_x) ** 2 + (y - gt_y) ** 2

        loss_factor = (torch.sum(mask != 0) + torch.sum(mask == 0)) / (
            torch.sum(mask != 0) + 1e-9
        )
        distance_loss = (
            loss_factor
            * (torch.log(distance + 1 + 1e-9) * mask).mean()
            * self.distance_weight
        )
        loss = distance_loss + visibility_loss
        return loss, {"distance": distance_loss, "visibility": visibility_loss}
