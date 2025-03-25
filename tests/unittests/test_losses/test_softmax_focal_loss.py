import torch
import torch.nn.functional as F

from luxonis_train.attached_modules.losses import SoftmaxFocalLoss


def test_softmax_focal_loss():
    predictions = torch.full((2, 3, 4, 4), 1.5)
    target_indices = torch.ones((2, 4, 4), dtype=torch.long)
    expected_loss = 0.0671

    targets = (
        F.one_hot(target_indices, num_classes=3).permute(0, 3, 1, 2).float()
    )

    loss_fn = SoftmaxFocalLoss(
        alpha=0.25, gamma=2.0, smooth=0.1, reduction="mean"
    )
    loss = loss_fn(predictions, targets)

    assert torch.isclose(loss, torch.tensor(expected_loss), atol=1e-4)
