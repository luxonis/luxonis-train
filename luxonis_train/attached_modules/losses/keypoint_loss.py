from typing import Any

import torch
from luxonis_ml.data import LabelType
from torch import Tensor

from luxonis_train.utils import (
    get_sigmas,
    get_with_default,
    process_keypoints_predictions,
)

from .base_loss import BaseLoss
from .bce_with_logits import BCEWithLogitsLoss


# TODO: Make it work on its own
class KeypointLoss(BaseLoss[Tensor, Tensor]):
    supported_labels = [LabelType.KEYPOINTS]

    def __init__(
        self,
        n_keypoints: int,
        bce_power: float = 1.0,
        sigmas: list[float] | None = None,
        area_factor: float | None = None,
        regression_loss_weight: float = 1.0,
        visibility_loss_weight: float = 1.0,
        **kwargs: Any,
    ):
        """Keypoint based loss that is computed from OKS-based
        regression and visibility loss.

        @type n_keypoints: int
        @param n_keypoints: Number of keypoints.
        @type bce_power: float
        @param bce_power: Power used for BCE visibility loss. Defaults
            to C{1.0}.
        @param sigmas: Sigmas used for OKS. If None then use COCO ones
            if possible or default ones. Defaults to C{None}.
        @type area_factor: float | None
        @param area_factor: Factor by which we multiply bbox area. If
            None then use default one. Defaults to C{None}.
        @type regression_loss_weight: float
        @param regression_loss_weight: Weight of regression loss.
            Defaults to C{1.0}.
        @type visibility_loss_weight: float
        @param visibility_loss_weight: Weight of visibility loss.
            Defaults to C{1.0}.
        """

        super().__init__(**kwargs)
        self.b_cross_entropy = BCEWithLogitsLoss(
            pos_weight=torch.tensor([bce_power]), **kwargs
        )
        self.sigmas = get_sigmas(sigmas, n_keypoints, caller_name=self.name)
        self.area_factor = get_with_default(
            area_factor, "bbox area scaling", self.name, default=0.53
        )
        self.regression_loss_weight = regression_loss_weight
        self.visibility_loss_weight = visibility_loss_weight

    def forward(
        self, prediction: Tensor, target: Tensor, area: Tensor
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Computes the keypoint loss and visibility loss for a given
        prediction and target.

        @type prediction: Tensor
        @param prediction: Predicted tensor of shape C{[n_detections,
            n_keypoints * 3]}.
        @type target: Tensor
        @param target: Target tensor of shape C{[n_detections,
            n_keypoints * 3]}.
        @type area: Tensor
        @param area: Area tensor of shape C{[n_detections]}.
        @rtype: tuple[Tensor, dict[str, Tensor]]
        @return: A tuple containing the total loss tensor of shape
            C{[1,]} and a dictionary with the regression loss and
            visibility loss tensors.
        """
        sigmas = self.sigmas.to(prediction.device)

        pred_x, pred_y, pred_v = process_keypoints_predictions(prediction)
        target_x = target[:, 0::3]
        target_y = target[:, 1::3]
        target_visibility = (target[:, 2::3] > 0).float()

        visibility_loss = (
            self.b_cross_entropy.forward(pred_v, target_visibility)
            * self.visibility_loss_weight
        )
        scales = area * self.area_factor

        distance = (target_x - pred_x) ** 2 + (target_y - pred_y) ** 2
        normalized_distance = (
            distance / (2 * sigmas**2) / (scales.view(-1, 1) + 1e-9) / 2
        )

        regression_loss = 1 - torch.exp(-normalized_distance)
        regression_loss = (regression_loss * target_visibility).sum(dim=1) / (
            target_visibility.sum(dim=1) + 1e-9
        )
        regression_loss = regression_loss.mean()
        regression_loss *= self.regression_loss_weight

        total_loss = regression_loss + visibility_loss

        return total_loss, {
            "kpt_regression": regression_loss,
            "kpt_visibility": visibility_loss,
        }
