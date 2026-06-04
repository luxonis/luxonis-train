import torch
from torch import Tensor, nn

from luxonis_train.nodes import OCRCTCHead

from .base_loss import BaseLoss


class CTCLoss(BaseLoss):
    """CTC loss with optional focal weighting for OCR.

    Metadata:
        - Module type: loss
        - Registry name: ``CTCLoss``
        - Task: OCR
        - Attached node types: ``OCRCTCHead``
        - Inputs: ``predictions``, ``target``
        - Outputs: scalar CTC loss

    Prediction format:
        ``predictions`` contains OCR logits shaped ``[B, T, C]``.

    Target format:
        ``target`` contains raw OCR labels encoded by the attached node encoder.

    Formula:
        Applies PyTorch ``nn.CTCLoss`` over log-softmax predictions and,
        optionally, focal weighting ``(1 - exp(-loss)) ** 2``.

    Provenance:
        - Source: Internal
        - License: Project license
        - Implementation notes: OCR task support is inferred from the attached
          ``OCRCTCHead`` rather than declared with ``supported_tasks``.

    """

    node: OCRCTCHead

    def __init__(self, use_focal_loss: bool = True, **kwargs):
        """Initialize the CTC loss with optional focal loss support.

        Args:
            use_focal_loss (bool): Whether to apply focal loss weighting to the CTC loss. Defaults to True.
            **kwargs (Any): Keyword arguments forwarded to the parent class.

        """
        super().__init__(**kwargs)
        self.loss_func = nn.CTCLoss(blank=0, reduction="none")
        self.use_focal_loss = use_focal_loss

    def forward(self, predictions: Tensor, target: Tensor) -> Tensor:
        """Compute the CTC loss, optionally applying focal loss.

        Args:
            predictions (Tensor): Network predictions of shape ``[B, T, C]``,
                where ``T`` is the sequence length, ``B`` is the batch size, and ``C`` is the number of classes.
            target (Tensor): Raw target sequences encoded by the node encoder.

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
