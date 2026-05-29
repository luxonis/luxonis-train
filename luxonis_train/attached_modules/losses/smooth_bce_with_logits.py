from typing import Literal

import torch
from torch import Tensor

from luxonis_train.tasks import Tasks

from .base_loss import BaseLoss
from .bce_with_logits import BCEWithLogitsLoss


class SmoothBCEWithLogitsLoss(BaseLoss):
    supported_tasks = [Tasks.SEGMENTATION, Tasks.CLASSIFICATION]

    def __init__(
        self,
        label_smoothing: float = 0.0,
        bce_pow: float = 1.0,
        weight: list[float] | None = None,
        reduction: Literal["mean", "sum", "none"] = "mean",
        **kwargs,
    ):
        """BCE with logits loss and label smoothing.

        Args:
            label_smoothing (float): Label smoothing factor. Defaults to ``0.0``.
            bce_pow (float): Weight for positive samples. Defaults to ``1.0``.
            weight (list[float] | None): a manual rescaling weight given to the loss of each batch
                element. If given, it has to be a list of length ``nbatch``.
            reduction (Literal["mean", "sum", "none"]): Specifies the reduction to apply to the
                output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be
                applied, ``'mean'``: the sum of the output will be divided by the number of
                elements in the output, ``'sum'``: the output will be summed. Note:
                ``size_average`` and ``reduce`` are in the process of being deprecated, and in the
                meantime, specifying either of those two args will override ``reduction``. Defaults
                to ``'mean'``.
        """
        super().__init__(**kwargs)
        self.positive_smooth_const = 1.0 - label_smoothing
        self.negative_smooth_const = label_smoothing
        self.criterion = BCEWithLogitsLoss(
            pos_weight=torch.tensor([bce_pow]),
            weight=weight,
            reduction=reduction,
        )

    def forward(self, predictions: Tensor, target: Tensor) -> Tensor:
        """Compute the BCE loss with label smoothing.

        Args:
            predictions (Tensor): Network predictions of shape (N, C, ...)
            target (Tensor): A tensor of the same shape as predictions.

        Returns:
            Tensor: A scalar tensor.
        """
        if predictions.shape != target.shape:
            raise RuntimeError(
                f"Target tensor dimension ({target.shape}) and predictions tensor "
                f"dimension ({predictions.shape}) should be the same."
            )

        if self.negative_smooth_const != 0.0:
            target = (
                target * self.positive_smooth_const
                + (1 - target) * self.negative_smooth_const
            )

        loss = self.criterion(predictions, target)
        assert isinstance(loss, Tensor)
        return loss
