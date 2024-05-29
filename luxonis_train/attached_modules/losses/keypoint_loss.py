from typing import Annotated

import torch
from pydantic import Field
from torch import Tensor

from luxonis_train.attached_modules.metrics.object_keypoint_similarity import (
    set_area_factor,
    set_sigmas,
)
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
        n_keypoints: int,
        bce_power: float = 1.0,
        sigmas: list[float] | None = None,
        area_factor: float | None = None,
        **kwargs,
    ):
        super().__init__(
            protocol=Protocol, required_labels=[LabelType.KEYPOINT], **kwargs
        )
        self.b_cross_entropy = BCEWithLogitsLoss(
            pos_weight=torch.tensor([bce_power]), **kwargs
        )
        self.sigmas = set_sigmas(
            sigmas=sigmas, n_keypoints=n_keypoints, class_name=self.__class__.__name__
        )
        self.area_factor = set_area_factor(
            area_factor, class_name=self.__class__.__name__
        )

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
        @param target: Target tensor of shape C{[n_detections, n_keypoints * 3]}.
        @type area: Tensor
        @param area: Area tensor of shape C{[n_detections]}.
        @rtype: tuple[Tensor, dict[str, Tensor]]
        @return: A tuple containing the total loss tensor of shape C{[1,]} and a
            dictionary with the regression loss and visibility loss tensors.
        """
        device = prediction.device
        sigmas = self.sigmas.to(device)

        pred_x, pred_y, pred_v = process_keypoints_predictions(prediction)
        gt_x = target[:, 0::3]
        gt_y = target[:, 1::3]
        gt_v = (target[:, 2::3] > 0).float()

        visibility_loss = self.b_cross_entropy.forward(pred_v, gt_v)
        scales = area * self.area_factor

        d = (gt_x - pred_x) ** 2 + (gt_y - pred_y) ** 2
        e = d / (2 * sigmas**2) / (scales.view(-1, 1) + 1e-9) / 2

        regression_loss_unreduced = 1 - torch.exp(-e)
        regression_loss_reduced = (regression_loss_unreduced * gt_v).sum(dim=1) / (
            gt_v.sum(dim=1) + 1e-9
        )
        regression_loss = regression_loss_reduced.mean()

        total_loss = regression_loss + visibility_loss

        return total_loss, {
            "regression": regression_loss,
            "visibility": visibility_loss,
        }
