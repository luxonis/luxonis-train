from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, amp

from luxonis_train.attached_modules.losses import BaseLoss
from luxonis_train.tasks import Tasks


class SoftmaxFocalLoss(BaseLoss):
    """Softmax focal loss for multiclass predictions.

    Metadata:
        - Module type: loss
        - Registry name: ``SoftmaxFocalLoss``
        - Task: SEGMENTATION, CLASSIFICATION
        - Attached node types: None
        - Inputs: ``predictions``, ``targets``
        - Outputs: scalar or unreduced softmax focal loss, depending on
          ``reduction``

    Prediction format:
        ``predictions`` contains multiclass logits with at least two channels.

    Target format:
        ``targets`` contains one-hot class targets with the same shape as
        ``predictions``.

    Formula:
        Applies softmax, optional label smoothing, alpha weighting, and focal
        modulation ``(1 - p_t) ** gamma``.

    Provenance:
        - Source: Internal
        - License: Project license
        - Implementation notes: Rejects one-channel binary inputs and keeps the
          focal computation in full precision under autocast.

    """

    supported_tasks = [Tasks.SEGMENTATION, Tasks.CLASSIFICATION]

    def __init__(
        self,
        alpha: float | list[float] = 0.25,
        gamma: float = 2.0,
        smooth: float = 0.0,
        reduction: Literal["none", "mean", "sum"] = "mean",
        **kwargs,
    ):
        """Compute focal loss for classification and segmentation with
        Softmax.

        Args:
            alpha (float | list[float]): Weighting factor for the rare class. Defaults to ``0.25``.
            gamma (float): Focusing parameter. Defaults to ``2.0``.
            smooth (float): Label smoothing factor. Defaults to ``0.0``.
            reduction (Literal["none", "mean", "sum"]): Reduction type. Defaults to ``"mean"``.
            **kwargs (Any): Keyword arguments forwarded to the parent class.

        """
        super().__init__(**kwargs)

        self.gamma = gamma
        self.smooth = smooth
        self.reduction = reduction

        if isinstance(alpha, list):
            self.alpha = torch.tensor(alpha)
        else:
            self.alpha = alpha

        if self.smooth is not None and not (0 <= self.smooth <= 1.0):
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

            if self.smooth:
                targets = targets.clamp(
                    self.smooth / (predictions.size(1) - 1),
                    1.0 - self.smooth,
                )

            pt = (targets * predictions).sum(dim=1) + self.smooth

            if isinstance(self.alpha, Tensor):
                if self.alpha.size(0) != predictions.size(1):
                    raise ValueError(
                        f"Alpha length {self.alpha.size(0)} does not "
                        f"match number of classes {predictions.size(1)}"
                    )
                alpha_t = self.alpha[targets.argmax(dim=1)]
            else:
                alpha_t = self.alpha

            pt = torch.as_tensor(pt, dtype=torch.float32)
            focal_term = torch.pow(1.0 - pt, self.gamma)
            loss = -alpha_t * focal_term * pt.log()

            if self.reduction == "mean":
                return loss.mean()
            if self.reduction == "sum":
                return loss.sum()
            return loss
