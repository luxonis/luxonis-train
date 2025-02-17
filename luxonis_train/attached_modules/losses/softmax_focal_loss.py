from typing import Literal

import torch
from torch import Tensor, amp
from torch.nn import functional as F

from luxonis_train.attached_modules.losses import BaseLoss
from luxonis_train.tasks import Tasks


# TODO: Add support for multi-class tasks
class SoftmaxFocalLoss(BaseLoss):
    supported_tasks = [Tasks.SEGMENTATION, Tasks.CLASSIFICATION]

    def __init__(
        self,
        alpha: float | list[float] = 0.25,
        gamma: float = 2.0,
        smooth: float = 0.0,
        reduction: Literal["none", "mean", "sum"] = "mean",
        **kwargs,
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

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        if predictions.shape != targets.shape:
            raise ValueError(
                f"Shape mismatch: {predictions.shape} vs {targets.shape}"
            )
        with amp.autocast(device_type=predictions.device.type, enabled=False):
            predictions = predictions.float()
            targets = targets.float()

            predictions = F.softmax(predictions, dim=1)

            if self.smooth:
                targets = torch.clamp(
                    targets,
                    self.smooth / (predictions.size(1) - 1),
                    1.0 - self.smooth,
                )

            pt = (targets * predictions).sum(dim=1) + self.smooth

            if isinstance(self.alpha, Tensor):
                if self.alpha.size(0) != predictions.size(1):
                    raise ValueError(
                        f"Alpha length {self.alpha.size(0)} does not match number of classes {predictions.size(1)}"
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
