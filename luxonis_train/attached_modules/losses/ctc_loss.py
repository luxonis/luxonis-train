from logging import getLogger
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from luxonis_train.utils import Labels, Packet

from .base_loss import BaseLoss

logger = getLogger(__name__)


class CTCLoss(BaseLoss[Tensor, Tensor, Tensor]):
    """CTC loss with optional focal loss weighting."""

    def __init__(self, use_focal_loss: bool = True, **kwargs: Any):
        """Initializes the CTC loss with optional focal loss support.

        @type use_focal_loss: bool
        @param use_focal_loss: Whether to apply focal loss weighting to
            the CTC loss. Defaults to True.
        """
        super().__init__(**kwargs)
        self.loss_func = nn.CTCLoss(blank=0, reduction="none")
        self.use_focal_loss = use_focal_loss

    def prepare(
        self, inputs: Packet[Tensor], labels: Labels
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Prepares the inputs, targets, and target lengths for loss
        computation.

        @type inputs: Packet[Tensor]
        @param inputs: A packet containing input tensors, typically
            network predictions.
        @type labels: Labels
        @param labels: A dictionary containing text labels and
            corresponding lengths.
        @rtype: tuple[Tensor, Tensor, Tensor]
        @return: A tuple of predictions, encoded targets, and target
            lengths.
        """
        preds = inputs["/classification"][0]
        targets = labels["/metadata/text"]
        target_lengths = torch.sum(targets != 0, dim=1)
        targets = self.node.encoder(targets).to(preds.device)  # type: ignore

        return preds, targets, target_lengths

    def forward(
        self, preds: Tensor, targets: Tensor, target_lengths: Tensor
    ) -> Tensor:
        """Computes the CTC loss, optionally applying focal loss.

        @type preds: Tensor
        @param preds: Network predictions of shape (B, T, C), where T is
            the sequence length, B is the batch size, and C is the
            number of classes.
        @type targets: Tensor
        @param targets: Encoded target sequences.
        @type target_lengths: Tensor
        @param target_lengths: Lengths of the target sequences.
        @rtype: Tensor
        @return: The computed loss as a scalar tensor.
        """

        preds = preds.permute(1, 0, 2)
        preds = preds.log_softmax(-1)

        T, B, _ = preds.shape
        preds_lengths = torch.full(
            (B,), T, dtype=torch.int64, device=preds.device
        )

        loss = self.loss_func(preds, targets, preds_lengths, target_lengths)

        if self.use_focal_loss:
            weight = (1.0 - torch.exp(-loss)) ** 2
            loss = loss * weight

        return loss.mean()
