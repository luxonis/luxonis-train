import torch

from luxonis_train.attached_modules.losses import (
    BCEWithLogitsLoss,
    CrossEntropyLoss,
    OHEMLoss,
)


def test_ohem_loss_cross_entropy():
    ohem_loss = OHEMLoss(CrossEntropyLoss, ohem_ratio=0.5, ohem_threshold=0.1)

    preds = torch.tensor([[1.0, 2.0], [1.0, 0.5]], requires_grad=True)
    target = torch.tensor([[1, 0], [0, 1]], dtype=torch.long)
    loss = ohem_loss(preds, target)
    assert loss.item() >= 0  # Loss should be non-negative


def test_ohem_loss_bce_with_logits():
    ohem_loss = OHEMLoss(BCEWithLogitsLoss, ohem_ratio=0.5, ohem_threshold=0.5)

    preds = torch.tensor([0.6, -0.3, 0.9], requires_grad=True)
    target = torch.tensor([1.0, 0.0, 1.0])
    loss = ohem_loss(preds, target)
    assert loss.item() >= 0  # Loss should be non-negative
