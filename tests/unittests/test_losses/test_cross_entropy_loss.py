import torch
import torch.nn.functional as F

from luxonis_train.attached_modules.losses import CrossEntropyLoss


def test_cross_entropy_loss():
    predictions = torch.full((2, 4), 1.0)
    target = torch.tensor([1, 2])
    loss_fn = CrossEntropyLoss(label_smoothing=0.1, reduction="mean")
    loss = loss_fn(predictions, target)
    expected = F.cross_entropy(
        predictions, target, label_smoothing=0.1, reduction="mean"
    )
    assert torch.isclose(loss, expected, atol=1e-4)
