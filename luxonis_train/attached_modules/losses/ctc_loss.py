import torch
from torch import Tensor, nn

from luxonis_train.nodes import OCRCTCHead

from .base_loss import BaseLoss


class CTCLoss(BaseLoss):
    """CTC loss with optional focal loss weighting."""

    node: OCRCTCHead

    def __init__(self, use_focal_loss: bool = True, **kwargs):
        """Initialize the CTC loss with optional focal loss support.

        Args:
            use_focal_loss (bool): Whether to apply focal loss weighting to the CTC loss. Defaults to True.
        """
        super().__init__(**kwargs)
        self.loss_func = nn.CTCLoss(blank=0, reduction="none")
        self.use_focal_loss = use_focal_loss

    def forward(self, predictions: Tensor, target: Tensor) -> Tensor:
        """Compute the CTC loss, optionally applying focal loss.

        Args:
            preds (Tensor): Network predictions of shape (B, T, C), where T is the sequence length, B is the batch size, and C is the number of classes.
            targets (Tensor): Encoded target sequences.

        Returns:
            Tensor: The computed loss as a scalar tensor.
        """
        target = self.node.encoder(target).to(predictions.device)
        target_lengths = torch.sum(target != 0, dim=1)

        predictions = predictions.permute(1, 0, 2)
        predictions = predictions.log_softmax(-1)

        T, B, _ = predictions.shape
        preds_lengths = torch.full(
            (B,), T, dtype=torch.int64, device=predictions.device
        )

        loss = self.loss_func(
            predictions, target, preds_lengths, target_lengths
        )

        if self.use_focal_loss:
            weight = (1.0 - torch.exp(-loss)) ** 2
            loss = loss * weight

        return loss.mean()
