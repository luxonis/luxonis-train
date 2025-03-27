from copy import deepcopy

import lightning.pytorch as pl
import pytest
import torch
from lightning.pytorch import LightningModule, Trainer
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from luxonis_train.callbacks.ema import EMACallback, ModelEma


class DummyModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(2, 2)

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = nn.MSELoss()(self(x), y)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


@pytest.fixture
def model() -> DummyModel:
    return DummyModel()


@pytest.fixture
def ema_callback() -> EMACallback:
    return EMACallback()


def test_ema_initialization(model: LightningModule, ema_callback: EMACallback):
    trainer = Trainer()
    ema_callback.on_fit_start(trainer, model)

    assert isinstance(ema_callback.ema, ModelEma)
    assert ema_callback.ema.decay == ema_callback.decay
    assert ema_callback.ema.use_dynamic_decay == ema_callback.use_dynamic_decay


def test_ema_update_on_batch_end(
    model: LightningModule, ema_callback: EMACallback
):
    trainer = Trainer()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    ema_callback.on_fit_start(trainer, model)

    initial_ema_state = {
        k: v.clone() for k, v in ema_callback.ema.state_dict_ema.items()
    }

    batch = torch.rand(2, 2)
    batch_idx = 0

    model.train()
    outputs = model(batch)
    model.zero_grad()
    outputs.sum().backward()
    optimizer.step()

    ema_callback.on_train_batch_end(trainer, model, outputs, batch, batch_idx)

    # Check that the EMA has been updated
    updated_state = ema_callback.ema.state_dict_ema
    assert any(
        not torch.equal(initial_ema_state[k], updated_state[k])
        for k in initial_ema_state
    )


def test_ema_state_saved_to_checkpoint(
    model: LightningModule, ema_callback: EMACallback
):
    trainer = Trainer()
    ema_callback.on_fit_start(trainer, model)

    checkpoint = {}
    ema_callback.on_save_checkpoint(trainer, model, checkpoint)

    assert "state_dict" in checkpoint


def test_load_from_checkpoint(
    model: LightningModule, ema_callback: EMACallback
):
    trainer = Trainer()

    checkpoint = {"state_dict": deepcopy(model.state_dict())}
    ema_callback.on_load_checkpoint(trainer, model, checkpoint)
    ema_callback.on_fit_start(trainer, model)
    assert (
        ema_callback.ema.state_dict_ema.keys()
        == checkpoint["state_dict"].keys()
    )


def test_validation_epoch_start_and_end(
    model: LightningModule, ema_callback: EMACallback
):
    trainer = Trainer()
    ema_callback.on_fit_start(trainer, model)

    ema_callback.on_validation_epoch_start(trainer, model)
    assert ema_callback.collected_state_dict is not None

    ema_callback.on_validation_end(trainer, model)
    for k in ema_callback.collected_state_dict:
        assert torch.equal(
            ema_callback.collected_state_dict[k], model.state_dict()[k]
        )


def test_ema_swapping_across_epochs(
    model: LightningModule, ema_callback: EMACallback
):
    class PreCheckCallback(pl.Callback):
        """Captures the training (original) weights before EMA swaps
        in."""

        def on_train_epoch_start(self, trainer, pl_module):
            pl_module.training_weights_on_train_epoch_start = {
                k: v.detach().clone()
                for k, v in pl_module.state_dict().items()
            }

        def on_validation_epoch_start(self, trainer, pl_module):
            pl_module.training_weights_on_val_start = {
                k: v.detach().clone()
                for k, v in pl_module.state_dict().items()
            }

        def on_save_checkpoint(self, trainer, pl_module, checkpoint):
            pl_module.training_weights_on_save_ckpt = {
                k: v.detach().clone()
                for k, v in pl_module.state_dict().items()
            }

    class PostCheckCallback(pl.Callback):
        """Verifies the model uses EMA weights during validation and on
        checkpoint save.

        It should revert to training weights on training epoch start.
        """

        def on_validation_epoch_start(self, trainer, pl_module):
            original_weights = pl_module.training_weights_on_val_start
            for k, v in pl_module.state_dict().items():
                # Should not match training weights because EMA has been swapped in
                assert not torch.equal(v, original_weights[k]), (
                    f"Param {k} still matches original weights - no EMA swap!"
                )

        def on_save_checkpoint(self, trainer, pl_module, checkpoint):
            # checkpoint["state_dict"] is the EMA weights
            original_weights_ckpt = pl_module.training_weights_on_save_ckpt
            for k, v in checkpoint["state_dict"].items():
                # Should differ from the training weights we captured
                assert not torch.equal(v, original_weights_ckpt[k]), (
                    f"Checkpoint param {k} matches original weights - no EMA in checkpoint!"
                )

        def on_train_epoch_start(self, trainer, pl_module):
            # Should be using original weights
            original_weights = pl_module.training_weights_on_train_epoch_start
            for k, v in pl_module.state_dict().items():
                assert torch.equal(v, original_weights[k]), (
                    f"Param {k} doesn't match original weights - EMA not swapped out!, epoch {trainer.current_epoch}"
                )

    x_train = torch.randn(50, 2)
    y_train = torch.randn(50, 2)
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=5)

    x_val = torch.randn(10, 2)
    y_val = torch.randn(10, 2)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=5)

    pre_callback = PreCheckCallback()
    post_callback = PostCheckCallback()

    trainer = Trainer(
        max_epochs=3,
        callbacks=[
            pre_callback,
            ema_callback,
            post_callback,
        ],  # The order matters: pre -> EMA -> post
        limit_val_batches=1,
    )

    trainer.fit(
        model, train_dataloaders=train_loader, val_dataloaders=val_loader
    )
