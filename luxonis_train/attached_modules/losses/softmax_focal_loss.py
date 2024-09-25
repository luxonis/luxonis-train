import logging
from typing import Any, Literal

import torch
from torch import Tensor

from luxonis_train.attached_modules.losses import BaseLoss
from luxonis_train.enums import TaskType

from .cross_entropy import CrossEntropyLoss

logger = logging.getLogger(__name__)


# TODO: Add support for multi-class tasks
class SoftmaxFocalLoss(BaseLoss[Tensor, Tensor]):
    supported_tasks: list[TaskType] = [
        TaskType.SEGMENTATION,
        TaskType.CLASSIFICATION,
    ]

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: Literal["none", "mean", "sum"] = "mean",
        **kwargs: Any,
    ):
        """Focal loss implementation for binary classification and
        segmentation tasks using Softmax.

        @type alpha: float
        @param alpha: Weighting factor for the rare class. Defaults to
            C{0.25}.
        @type gamma: float
        @param gamma: Focusing parameter. Defaults to C{2.0}.
        @type reduction: Literal["none", "mean", "sum"]
        @param reduction: Reduction type. Defaults to C{"mean"}.
        """
        super().__init__(**kwargs)

        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce_criterion = CrossEntropyLoss(reduction="none", **kwargs)

    def forward(self, predictions: Tensor, target: Tensor) -> Tensor:
        ce_loss = self.ce_criterion.forward(predictions, target)
        pt = torch.exp(-ce_loss)
        loss = ce_loss * ((1 - pt) ** self.gamma) * self.alpha

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss
