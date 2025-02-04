import pytest
import torch
from torch import Tensor

from luxonis_train.attached_modules.losses import BCEWithLogitsLoss

torch.manual_seed(42)


def test_forward_pass():
    batch_sizes = [1, 2, 10, 11, 15, 64, 128, 255]
    n_classes = [1, 2, 3, 4, 64]

    for bs in batch_sizes:
        for n_cl in n_classes:
            targets = torch.ones([bs, n_cl], dtype=torch.float32)
            predictions = torch.full([bs, n_cl], 1.5)  # logit
            loss_fn = BCEWithLogitsLoss()

            loss = loss_fn.forward(
                predictions, targets
            )  # -log(sigmoid(1.5)) = 0.2014

            assert isinstance(loss, Tensor)
            assert loss.shape == torch.Size([])
            assert torch.round(loss, decimals=2) == 0.20


def test_minimum():
    bs, n_classes = 10, 4

    targets = torch.ones([bs, n_classes], dtype=torch.float32)
    predictions = torch.full([bs, n_classes], 10e3)  # logit
    loss_fn = BCEWithLogitsLoss()

    loss = loss_fn.forward(predictions, targets)
    assert torch.round(loss, decimals=2) == 0.0

    targets = torch.zeros([bs, n_classes], dtype=torch.float32)
    predictions = torch.full([bs, n_classes], -10e3)  # logit
    loss_fn = BCEWithLogitsLoss()

    loss = loss_fn.forward(predictions, targets)
    assert torch.round(loss, decimals=2) == 0.0


def test_weights():
    bs, n_classes = 10, 4

    targets = torch.ones([bs, n_classes], dtype=torch.float32)
    predictions = torch.rand([bs, n_classes]) * 10 - 5  # logit

    loss_fn_weight = BCEWithLogitsLoss(
        pos_weight=torch.randint(1, 10, torch.Size((n_classes,)))
    )
    loss_fn_no_weight = BCEWithLogitsLoss()

    loss_weight = loss_fn_weight.forward(predictions, targets)
    loss_no_weight = loss_fn_no_weight.forward(predictions, targets)
    assert loss_weight != loss_no_weight


def test_invalid():
    loss_fn = BCEWithLogitsLoss()
    with pytest.raises(RuntimeError):
        loss_fn.forward(torch.rand(10, 10), torch.rand(15, 15))
