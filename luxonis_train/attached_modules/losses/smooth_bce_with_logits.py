from typing import Literal

import torch
from torch import Tensor

from .base_loss import BaseLoss
from .bce_with_logits import BCEWithLogitsLoss


class SmoothBCEWithLogitsLoss(BaseLoss[list[Tensor], Tensor]):
    def __init__(
        self,
        label_smoothing: float = 0.0,
        bce_pow: float = 1.0,
        weight: list[float] | None = None,
        reduction: Literal["mean", "sum", "none"] = "mean",
        **kwargs,
    ):
        """BCE with logits loss and label smoothing.

        @type label_smoothing: float
        @param label_smoothing: Label smoothing factor. Defaults to C{0.0}.
        @type bce_pow: float
        @param bce_pow: Weight for positive samples. Defaults to C{1.0}.
        @type weight: list[float] | None
        @param weight: a manual rescaling weight given to the loss of each batch
            element. If given, it has to be a list of length C{nbatch}.
        @type reduction: Literal["mean", "sum", "none"]
        @param reduction: Specifies the reduction to apply to the output: C{'none'} |
            C{'mean'} | C{'sum'}. C{'none'}: no reduction will be applied, C{'mean'}:
            the sum of the output will be divided by the number of elements in the
            output, C{'sum'}: the output will be summed. Note: C{size_average} and
            C{reduce} are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override C{reduction}. Defaults to
            C{'mean'}.
        @type kwargs: dict
        @param kwargs: Additional arguments to pass to L{BaseLoss}.
        """
        super().__init__(**kwargs)
        self.negative_smooth_const = 1.0 - 0.5 * label_smoothing
        self.positive_smooth_const = 0.5 * label_smoothing
        self.criterion = BCEWithLogitsLoss(
            node=self.node,
            pos_weight=torch.tensor(
                [bce_pow],
            ),
            weight=weight,
            reduction=reduction,
        )

    def forward(self, predictions: list[Tensor], target: Tensor) -> Tensor:
        """Computes the BCE loss with label smoothing.

        @type predictions: list[Tensor]
        @param predictions: List of tensors of shape (N, n_classes), containing the
            predicted class scores.
        @type target: Tensor
        @param target: A tensor of shape (N,), containing the ground-truth class labels
        @rtype: Tensor
        @return: A scalar tensor.
        """
        prediction = predictions[0]
        smoothed_target = torch.full_like(
            prediction,
            self.negative_smooth_const,
            device=prediction.device,
        )
        smoothed_target[torch.arange(len(target)), target] = self.positive_smooth_const
        return self.criterion.forward(prediction, smoothed_target)
