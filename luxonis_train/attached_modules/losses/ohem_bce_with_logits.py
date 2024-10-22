from typing import Any

from .ohem_loss import OHEMLoss
from .bce_with_logits import BCEWithLogitsLoss


class OHEMBCEWithLogitsLoss(OHEMLoss):
    """This criterion computes the binary cross entropy loss between input
    logits and target with OHEM (Online Hard Example Mining)."""

    def __init__(
        self,
        **kwargs: Any,
    ):
        kwargs.update(criterion=BCEWithLogitsLoss)
        super().__init__(**kwargs)

