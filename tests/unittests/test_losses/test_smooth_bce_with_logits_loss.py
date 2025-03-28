import torch

from luxonis_train.attached_modules.losses import SmoothBCEWithLogitsLoss


def test_smooght_bce_with_logits():
    predictions = torch.full((2, 1, 4, 4), 0.5)
    targets = torch.ones((2, 1, 4, 4), dtype=torch.float32)
    expected_loss = 0.4741

    loss_fn = SmoothBCEWithLogitsLoss()
    loss = loss_fn(predictions, targets)

    assert torch.isclose(loss, torch.tensor(expected_loss), atol=1e-4)
