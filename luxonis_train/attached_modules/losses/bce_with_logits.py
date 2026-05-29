from typing import Literal

import torch
from torch import Tensor, nn

from luxonis_train.tasks import Tasks

from .base_loss import BaseLoss


class BCEWithLogitsLoss(BaseLoss):
    """Combines a `nn.Sigmoid` layer and the `nn.BCELoss` in one
    single class.

    This version is more numerically stable than using a plain
    ``Sigmoid`` followed by a {BCELoss} as, by combining the operations
    into one layer, we take advantage of the log-sum-exp trick for
    numerical stability.

    """

    supported_tasks = [Tasks.SEGMENTATION, Tasks.CLASSIFICATION]

    def __init__(
        self,
        weight: list[float] | None = None,
        reduction: Literal["none", "mean", "sum"] = "mean",
        pos_weight: Tensor | None = None,
        **kwargs,
    ):
        """
        Args:
            weight (list[float] | None): a manual rescaling weight given to the loss of each batch element. If given, has to be a list of length ``nbatch``. Defaults to ``None``.
            reduction (Literal["none", "mean", "sum"]): Specifies the reduction to apply to the output: ``"none"`` | ``"mean"`` | ``"sum"``. ``"none"``: no reduction will be applied, ``"mean"``: the sum of the output will be divided by the number of elements in the output, ``"sum"``: the output will be summed. Note: ``size_average`` and ``reduce`` are in the process of being deprecated, and in the meantime, specifying either of those two args will override ``reduction``. Defaults to ``"mean"``.
            pos_weight (Tensor | None): a weight of positive examples to be broadcasted with target. Must be a tensor with equal size along the class dimension to the number of classes. Pay close attention to PyTorch's broadcasting semantics in order to achieve the desired operations. For a target of size [B, C, H, W] (where B is batch size) pos_weight of size [B, C, H, W] will apply different pos_weights to each element of the batch or [C, H, W] the same pos_weights across the batch. To apply the same positive weight along all spacial dimensions for a 2D multi-class target [C, H, W] use: [C, 1, 1]. Defaults to ``None``.
        """
        super().__init__(**kwargs)
        self.criterion = nn.BCEWithLogitsLoss(
            weight=(torch.tensor(weight) if weight is not None else None),
            reduction=reduction,
            pos_weight=pos_weight if pos_weight is not None else None,
        )

    def forward(self, predictions: Tensor, target: Tensor) -> Tensor:
        """Compute the BCE loss from logits.

        Args:
            predictions (Tensor): Network predictions of shape (N, C, ...)
            target (Tensor): A tensor of the same shape as predictions.

        Returns:
            Tensor: A scalar tensor.
        """
        if predictions.shape != target.shape:
            raise RuntimeError(
                f"Target tensor dimension ({target.shape}) and preds tensor "
                f"dimension ({predictions.shape}) should be the same."
            )
        return self.criterion(predictions, target)
