import lightning.pytorch as pl
import pytest
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from luxonis_train.callbacks.fail_on_no_train_batches import (
    FailOnNoTrainBatches,
    _format_details,
    _minimum_batch_count,
    _minimum_required_size,
)


def test_minimum_batch_count_for_int_limit():
    assert _minimum_batch_count(5) == 1
    assert _minimum_batch_count(0) is None


def test_minimum_batch_count_for_float_limit():
    assert _minimum_batch_count(0.25) == 4
    assert _minimum_batch_count(0.0) is None


def test_minimum_required_size_with_drop_last():
    batch_size, world_size = 8, 2
    assert _minimum_required_size(batch_size, True, world_size, 1.0) == 16


def test_minimum_required_size_without_drop_last():
    # ceil(1 / 0.5) = 2 batches -> (2 - 1) * 8 * 2 + 1
    batch_size, world_size = 8, 2
    assert _minimum_required_size(batch_size, False, world_size, 0.5) == 17


def test_minimum_required_size_needs_batch_size_and_drop_last():
    assert _minimum_required_size(None, True, 1, 1.0) is None
    assert _minimum_required_size(8, None, 1, 1.0) is None


def test_format_details_renders_all_parts():
    message = _format_details(3, 8, 8, 1, True, 1.0)
    assert message == (
        "(details: dataset_size=3, min_required_size=8, missing=5; "
        "params: batch_size=8, world_size=1, drop_last=True, "
        "limit_train_batches=1.0)"
    )


def test_format_details_skips_unknown_parts():
    message = _format_details(None, None, None, 2, None, 0.5)
    assert message == (
        "(details: ; params: world_size=2, limit_train_batches=0.5)"
    )


def test_fit_fails_when_drop_last_removes_the_only_batch():
    loader = DataLoader(
        TensorDataset(torch.zeros(3, 1)), batch_size=8, drop_last=True
    )
    trainer = pl.Trainer(
        callbacks=[FailOnNoTrainBatches()],
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        accelerator="cpu",
    )
    with pytest.raises(RuntimeError) as exc_info:
        trainer.fit(_TinyModule(), loader)

    assert str(exc_info.value) == (
        "No training batches found. Your dataset is smaller than the "
        "effective batch size or skip_last_batch=True removed the last "
        "batch. (details: dataset_size=3, min_required_size=8, missing=5; "
        "params: batch_size=8, world_size=1, drop_last=True, "
        "limit_train_batches=1.0)"
    )


class _TinyModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(1, 1)

    def training_step(self, batch: list[Tensor]) -> Tensor: ...

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.parameters(), lr=0.1)
