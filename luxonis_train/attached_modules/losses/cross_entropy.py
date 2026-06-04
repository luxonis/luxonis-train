from typing import Literal

import torch
from loguru import logger
from torch import Tensor, nn

from luxonis_train.tasks import Tasks

from .base_loss import BaseLoss


class CrossEntropyLoss(BaseLoss):
    """Compute cross entropy between logits and class targets.

    Metadata:
        - Module type: loss
        - Registry name: ``CrossEntropyLoss``
        - Task: SEGMENTATION, CLASSIFICATION
        - Attached node types: None
        - Inputs: ``predictions``, ``target``
        - Outputs: scalar cross entropy loss

    Prediction format:
        ``predictions`` contains class logits. If a one-channel tensor is
        provided for a binary task, a dummy channel is added before computing
        the loss.

    Target format:
        ``target`` may contain class indices or one-hot class labels. One-hot
        targets are converted with ``argmax`` along the channel dimension.

    Formula:
        Applies PyTorch ``nn.CrossEntropyLoss`` after target normalization.

    Provenance:
        - Source: PyTorch
        - License: Project license
        - Implementation notes: Handles one-hot targets and logs a warning
          once when one-channel logits are adapted for cross entropy.

    """

    supported_tasks = [Tasks.SEGMENTATION, Tasks.CLASSIFICATION]

    def __init__(
        self,
        weight: list[float] | None = None,
        ignore_index: int = -100,
        reduction: Literal["none", "mean", "sum"] = "mean",
        label_smoothing: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.criterion = nn.CrossEntropyLoss(
            weight=(torch.tensor(weight) if weight is not None else None),
            ignore_index=ignore_index,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )
        self._was_logged = False

    def forward(self, predictions: Tensor, target: Tensor) -> Tensor:
        if predictions.ndim == target.ndim:
            ch_dim = 1 if predictions.ndim > 1 else 0
            if predictions.shape[ch_dim] == 1:
                if not self._was_logged:
                    logger.warning(
                        "`CrossEntropyLoss` expects at least 2 classes. "
                        "Attempting to fix by adding a dummy channel. "
                        "If you want to be sure, use `BCEWithLogitsLoss` instead."
                    )
                    self._was_logged = True
                predictions = torch.cat(
                    [torch.zeros_like(predictions), predictions], dim=ch_dim
                )
                if target.shape[ch_dim] == 1:
                    target = torch.cat([1 - target, target], dim=ch_dim)
            target = target.argmax(dim=ch_dim)

        if target.ndim != predictions.ndim - 1:
            raise RuntimeError(
                f"Target tensor dimension should equeal to preds dimension - 1 ({predictions.ndim - 1}) "
                f"but is ({target.ndim})."
            )
        return self.criterion(predictions, target)
