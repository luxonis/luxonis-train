from typing import Literal

import pytest
import torch
import torch.nn.functional as F

from luxonis_train.attached_modules.losses import SoftmaxFocalLoss


@pytest.mark.parametrize(
    ("reduction", "expected_loss"),
    [
        ("mean", 0.0671),
        ("sum", 0.5371),
        (
            "none",
            [
                [[0.0671, 0.0671], [0.0671, 0.0671]],
                [[0.0671, 0.0671], [0.0671, 0.0671]],
            ],
        ),
    ],
)
def test_softmax_focal_loss(
    reduction: Literal["none", "mean", "sum"],
    expected_loss: float | list[list[list[float]]],
):
    predictions = torch.full((2, 3, 2, 2), 1.5)
    target_indices = torch.ones((2, 2, 2), dtype=torch.long)

    targets = (
        F.one_hot(target_indices, num_classes=3).permute(0, 3, 1, 2).float()
    )

    loss_fn = SoftmaxFocalLoss(
        alpha=0.25, gamma=2.0, smooth=0.1, reduction=reduction
    )
    loss = loss_fn(predictions, targets)

    assert torch.isclose(loss, torch.tensor(expected_loss), atol=1e-4).all()


def test_softmax_focal_loss_shape_mismatch():
    predictions = torch.full((2, 3, 3, 2), 1.5)
    target_indices = torch.ones((2, 2, 2), dtype=torch.long)

    targets = (
        F.one_hot(target_indices, num_classes=3).permute(0, 3, 1, 2).float()
    )

    loss_fn = SoftmaxFocalLoss(alpha=0.25, gamma=2.0, smooth=0.1)
    with pytest.raises(ValueError, match="Shape mismatch"):
        loss_fn(predictions, targets)


def test_softmax_focal_loss_binary():
    predictions = torch.full((2, 1, 2, 2), 1.5)
    target_indices = torch.ones((2, 2, 2), dtype=torch.long)

    targets = (
        F.one_hot(target_indices, num_classes=2).permute(0, 3, 1, 2).float()
    )

    loss_fn = SoftmaxFocalLoss(alpha=0.25, gamma=2.0, smooth=0.1)
    with pytest.raises(ValueError, match="binary tasks"):
        loss_fn(predictions, targets)
