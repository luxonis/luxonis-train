from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, amp

from luxonis_train.attached_modules.losses import BaseLoss
from luxonis_train.tasks import Tasks


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

        self._gamma = gamma
        self._smooth = smooth
        self._reduction = reduction

        if isinstance(alpha, list):
            self._alpha = torch.tensor(alpha)
        else:
            self._alpha = alpha

        if self._smooth is not None and not (0 <= self._smooth <= 1.0):
            raise ValueError("smooth value should be in [0,1]")

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        if predictions.size(1) < 2:
            raise ValueError(
                "SoftmaxFocalLoss is not suitable for binary tasks. "
                "Please use SigmoidFocalLoss instead."
            )

        if predictions.shape != targets.shape:
            raise ValueError(
                f"Shape mismatch: {predictions.shape} vs {targets.shape}"
            )
        with amp.autocast(device_type=predictions.device.type, enabled=False):
            predictions = predictions.float()
            targets = targets.float()

            predictions = F.softmax(predictions, dim=1)

            if self._smooth:
                targets = targets.clamp(
                    self._smooth / (predictions.size(1) - 1),
                    1.0 - self._smooth,
                )

            pt = (targets * predictions).sum(dim=1) + self._smooth

            if isinstance(self._alpha, Tensor):
                if self._alpha.size(0) != predictions.size(1):
                    raise ValueError(
                        f"Alpha length {self._alpha.size(0)} does not "
                        f"match number of classes {predictions.size(1)}"
                    )
                alpha_t = self._alpha[targets.argmax(dim=1)]
            else:
                alpha_t = self._alpha

            pt = torch.as_tensor(pt, dtype=torch.float32)
            focal_term = torch.pow(1.0 - pt, self._gamma)
            loss = -alpha_t * focal_term * pt.log()

            if self._reduction == "mean":
                return loss.mean()
            if self._reduction == "sum":
                return loss.sum()
            return loss
