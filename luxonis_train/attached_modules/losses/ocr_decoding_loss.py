import torch
from torch import Tensor, nn

from .base_loss import BaseLoss


class FocalCTC(BaseLoss[Tensor, Tensor]):
    def __init__(self, blank=0, alpha=0.99, gamma=1.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.loss = nn.CTCLoss(zero_infinity=True, blank=blank, reduction="none")

    def forward(
            self,
            logits,
            labels
    ):
        input_lengths = torch.full(size=(logits.shape[1],), fill_value=logits.shape[0], dtype=torch.long)

        targets, target_lengths, max_len = labels

        ctc_loss = self.loss(logits, targets, input_lengths, target_lengths)
        p = torch.exp(-ctc_loss)
        focal_ctc_loss = (self.alpha * ((1 - p) ** self.gamma) * ctc_loss)
        focal_ctc_loss = focal_ctc_loss.mean()

        return focal_ctc_loss


class SmoothCTCLoss(BaseLoss[Tensor, Tensor, Tensor, Tensor]):

    def __init__(self, num_classes, blank=0, weight=0.01):
        super().__init__()
        self.weight = weight
        self.num_classes = num_classes

        self.ctc = nn.CTCLoss(reduction='mean', blank=blank, zero_infinity=True)
        self.kldiv = nn.KLDivLoss(reduction='batchmean')

    def forward(self, log_probs, targets, input_lengths, target_lengths):
        ctc_loss = self.ctc(log_probs, targets, input_lengths, target_lengths)

        kl_inp = log_probs.transpose(0, 1)
        kl_tar = torch.full_like(kl_inp, 1. / self.num_classes)
        kldiv_loss = self.kldiv(kl_inp, kl_tar)

        loss = (1. - self.weight) * ctc_loss + self.weight * kldiv_loss
        return loss.mean()
