from typing import Any

from .ohem_loss import OHEMLoss
from .cross_entropy import CrossEntropyLoss


class OHEMCrossEntropyLoss(OHEMLoss):
    """This criterion computes the cross entropy loss between input
    logits and target with OHEM (Online Hard Example Mining)."""

    def __init__(
        self,
        **kwargs: Any,
    ):
        kwargs.update(criterion=CrossEntropyLoss)
        super().__init__(**kwargs)

