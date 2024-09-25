from logging import getLogger
from typing import Any, Literal

import torch
import torch.nn as nn
from torch import Tensor

from luxonis_train.enums import TaskType

from .base_loss import BaseLoss

logger = getLogger(__name__)


class CrossEntropyLoss(BaseLoss[Tensor, Tensor]):
    """This criterion computes the cross entropy loss between input
    logits and target."""

    supported_tasks: list[TaskType] = [
        TaskType.SEGMENTATION,
        TaskType.CLASSIFICATION,
    ]

    def __init__(
        self,
        weight: list[float] | None = None,
        ignore_index: int = -100,
        reduction: Literal["none", "mean", "sum"] = "mean",
        label_smoothing: float = 0.0,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        self.criterion = nn.CrossEntropyLoss(
            weight=(torch.tensor(weight) if weight is not None else None),
            ignore_index=ignore_index,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )
        self._was_logged = False

    def forward(self, preds: Tensor, target: Tensor) -> Tensor:
        if preds.ndim == target.ndim:
            ch_dim = 1 if preds.ndim > 1 else 0
            if preds.shape[ch_dim] == 1:
                if not self._was_logged:
                    logger.warning(
                        "`CrossEntropyLoss` expects at least 2 classes. "
                        "Attempting to fix by adding a dummy channel. "
                        "If you want to be sure, use `BCEWithLogitsLoss` instead."
                    )
                    self._was_logged = True
                preds = torch.cat([torch.zeros_like(preds), preds], dim=ch_dim)
                if target.shape[ch_dim] == 1:
                    target = torch.cat([1 - target, target], dim=ch_dim)
            target = target.argmax(dim=ch_dim)

        if target.ndim != preds.ndim - 1:
            raise RuntimeError(
                f"Target tensor dimension should equeal to preds dimension - 1 ({preds.ndim-1}) "
                f"but is ({target.ndim})."
            )
        return self.criterion(preds, target)
