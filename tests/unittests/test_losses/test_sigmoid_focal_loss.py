import torch

from luxonis_train.attached_modules.losses import SigmoidFocalLoss


def test_sigmoid_focal_loss():
    predictions = torch.full((2, 1, 4, 4), -1.0)
    targets = torch.zeros((2, 1, 4, 4), dtype=torch.float32)
    expected_loss = 0.0170

    loss_fn = SigmoidFocalLoss(alpha=0.25, gamma=2.0, reduction="mean")
    loss = loss_fn(predictions, targets)

    assert torch.isclose(loss, torch.tensor(expected_loss), atol=1e-4)
