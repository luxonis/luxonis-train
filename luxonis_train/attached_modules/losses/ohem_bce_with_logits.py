from .bce_with_logits import BCEWithLogitsLoss
from .ohem_loss import OHEMLoss


class OHEMBCEWithLogitsLoss(OHEMLoss):
    """This criterion computes the binary cross entropy loss between
    input logits and target with OHEM (Online Hard Example Mining)."""

    def __init__(self, **kwargs):
        kwargs.update(criterion=BCEWithLogitsLoss)
        super().__init__(**kwargs)
