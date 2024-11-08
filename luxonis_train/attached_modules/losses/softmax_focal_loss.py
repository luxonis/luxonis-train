import logging
from typing import Any, Literal

import torch
from torch import Tensor
from torch.nn import functional as F

from luxonis_train.attached_modules.losses import BaseLoss
from luxonis_train.enums import TaskType

logger = logging.getLogger(__name__)


# TODO: Add support for multi-class tasks
class SoftmaxFocalLoss(BaseLoss[Tensor, Tensor]):
    supported_tasks: list[TaskType] = [
        TaskType.SEGMENTATION,
        TaskType.CLASSIFICATION,
    ]

    def __init__(
        self,
        alpha: float | list[float] = 0.25,
        gamma: float = 2.0,
        smooth: float = 0.0,
        reduction: Literal["none", "mean", "sum"] = "mean",
        **kwargs: Any,
    ):
        """Focal loss implementation for classification and segmentation
        tasks using Softmax.

        @type alpha: float | list[float]
        @param alpha: Weighting factor for the rare class. Defaults to
            C{0.25}.
        @type gamma: float
        @param gamma: Focusing parameter. Defaults to C{2.0}.
        @type smooth: float
        @param smooth: Label smoothing factor. Defaults to C{0.0}.
        @type reduction: Literal["none", "mean", "sum"]
        @param reduction: Reduction type. Defaults to C{"mean"}.
        """
        super().__init__(**kwargs)

        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.reduction = reduction

        if isinstance(alpha, list):
            self.alpha = torch.tensor(alpha)
        else:
            self.alpha = alpha

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError("smooth value should be in [0,1]")

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        if logits.shape != targets.shape:
            raise ValueError(
                f"Shape mismatch: {logits.shape} vs {targets.shape}"
            )
        logits = F.softmax(logits, dim=1)

        if self.smooth:
            targets = torch.clamp(
                targets, self.smooth / (logits.size(1) - 1), 1.0 - self.smooth
            )

        pt = (targets * logits).sum(dim=1) + self.smooth

        if isinstance(self.alpha, torch.Tensor):
            if self.alpha.size(0) != logits.size(1):
                raise ValueError(
                    f"Alpha length {self.alpha.size(0)} does not match number of classes {logits.size(1)}"
                )
            alpha_t = self.alpha[targets.argmax(dim=1)]
        else:
            alpha_t = self.alpha

        pt = torch.as_tensor(pt, dtype=torch.float32)
        focal_term = torch.pow(1.0 - pt, self.gamma)
        loss = -alpha_t * focal_term * pt.log()  # type: ignore

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
