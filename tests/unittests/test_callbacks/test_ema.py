import shutil
from copy import deepcopy

import lightning.pytorch as pl
import pytest
import torch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from luxonis_train.callbacks.ema import EMACallback, ModelEma


class DummyModel(pl.LightningModule):
    training_weights_on_train_epoch_start: dict[str, Tensor]
    training_weights_on_val_start: dict[str, Tensor]
    training_weights_on_save_ckpt: dict[str, Tensor]

    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(2, 2)

    def forward(self, x: Tensor) -> Tensor:
        return self.layer(x)

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int):
        x, y = batch
        return nn.MSELoss()(self(x), y)

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int):
        x, y = batch
        loss = nn.MSELoss()(self(x), y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_metric", -loss, prog_bar=True)

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

    # Slightly modify the EMA state dict to simulate a different state
    for param_key in ema_callback.ema.state_dict_ema:
        ema_callback.ema.state_dict_ema[param_key] += 0.1 * torch.randn_like(
            ema_callback.ema.state_dict_ema[param_key]
        )

    ema_callback.on_validation_epoch_start(trainer, model)
    assert ema_callback.collected_state_dict is not None

    collected_state = model.state_dict()
    ema_callback.on_validation_end(trainer, model)

    diffs = sum(
        not torch.equal(collected_state[p], ema_callback.ema.state_dict_ema[p])
        for p in collected_state
    )
    assert diffs > 0, "Parameters did not swap after on_validation_end!"


def test_ema_swapping_across_training(
    model: LightningModule, ema_callback: EMACallback
):
    class PreCheckCallback(pl.Callback):
        """Captures the training (original) weights before EMA swaps
        in."""

        def on_train_epoch_start(
            self, trainer: pl.Trainer, pl_module: DummyModel
        ) -> None:
            pl_module.training_weights_on_train_epoch_start = {
                k: v.detach().clone()
                for k, v in pl_module.state_dict().items()
            }

        def on_validation_epoch_start(
            self, trainer: pl.Trainer, pl_module: DummyModel
        ) -> None:
            pl_module.training_weights_on_val_start = {
                k: v.detach().clone()
                for k, v in pl_module.state_dict().items()
            }

        def on_save_checkpoint(
            self,
            trainer: pl.Trainer,
            pl_module: DummyModel,
            checkpoint: dict[str, Tensor],
        ) -> None:
            pl_module.training_weights_on_save_ckpt = {
                k: v.detach().clone()
                for k, v in pl_module.state_dict().items()
            }

    class PostCheckCallback(pl.Callback):
        """Verifies the model uses EMA weights during validation and on
        checkpoint save.

        It should revert to training weights on training epoch start.
        """

        def on_train_epoch_start(
            self, trainer: pl.Trainer, pl_module: DummyModel
        ) -> None:
            original_weights = pl_module.training_weights_on_train_epoch_start
            diffs = sum(
                not torch.equal(pl_module.state_dict()[k], original_weights[k])
                for k in pl_module.state_dict()
            )
            assert diffs == 0, "Parameters changed after on_train_epoch_start!"

        def on_validation_epoch_start(
            self, trainer: pl.Trainer, pl_module: DummyModel
        ) -> None:
            original_weights = pl_module.training_weights_on_val_start
            diffs = sum(
                not torch.equal(pl_module.state_dict()[k], original_weights[k])
                for k in pl_module.state_dict()
            )
            assert diffs > 0, (
                "Parameters did not swap after on_validation_epoch_start!"
            )

        def on_save_checkpoint(
            self,
            trainer: pl.Trainer,
            pl_module: DummyModel,
            checkpoint: dict[str, Tensor],
        ) -> None:
            original_weights = pl_module.training_weights_on_save_ckpt
            diffs = sum(
                not torch.equal(pl_module.state_dict()[k], original_weights[k])
                for k in pl_module.state_dict()
            )
            assert diffs == 0, "Parameters changed after on_save_checkpoint!"

    x_train = torch.randn(50, 2)
    y_train = torch.randn(50, 2)
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=5)

    x_val = torch.randn(10, 2)
    y_val = torch.randn(10, 2)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=5)

    pre_callback = PreCheckCallback()
    post_callback = PostCheckCallback()

    # Simulate luxonis-train 2 ModelCheckpoint callbacks
    checkpoint_min = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        filename="min-loss",
        save_top_k=1,
    )
    checkpoint_best = ModelCheckpoint(
        monitor="val_metric",
        mode="max",
        filename="best-metric",
        save_top_k=1,
    )

    trainer = Trainer(
        max_epochs=4,
        check_val_every_n_epoch=2,
        callbacks=[
            pre_callback,
            ema_callback,
            post_callback,
            checkpoint_min,
            checkpoint_best,
        ],
        limit_val_batches=1,
        num_sanity_val_steps=0,
        default_root_dir="test_ema_swapping_logs",
    )

    trainer.fit(
        model, train_dataloaders=train_loader, val_dataloaders=val_loader
    )

    shutil.rmtree("test_ema_swapping_logs", ignore_errors=True)
