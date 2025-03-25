import torch
import torch.nn.functional as F

from luxonis_train.attached_modules.losses import SmoothBCEWithLogitsLoss


def test_smooth_bce_with_logits_loss():
    predictions = torch.full((2, 3), 0.5)
    targets = torch.ones((2, 3))
    loss_fn = SmoothBCEWithLogitsLoss(label_smoothing=0.1, reduction="mean")
    loss = loss_fn(predictions, targets)
    smoothed_targets = targets * (1 - 0.1) + 0.1 / targets.size(1)
    expected = F.binary_cross_entropy_with_logits(
        predictions, smoothed_targets, reduction="mean"
    )
    assert torch.isclose(loss, expected, atol=1e-4)
