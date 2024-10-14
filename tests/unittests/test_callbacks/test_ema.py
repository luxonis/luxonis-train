from copy import deepcopy

import pytest
import torch
from pytorch_lightning import LightningModule, Trainer

from luxonis_train.callbacks.ema import EMACallback, ModelEma


class SimpleModel(LightningModule):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.layer = torch.nn.Linear(2, 2)

    def forward(self, x):
        return self.layer(x)


@pytest.fixture
def model():
    return SimpleModel()


@pytest.fixture
def ema_callback():
    return EMACallback()


def test_ema_initialization(model, ema_callback):
    trainer = Trainer()
    ema_callback.on_fit_start(trainer, model)

    assert isinstance(ema_callback.ema, ModelEma)
    assert ema_callback.ema.decay == ema_callback.decay
    assert ema_callback.ema.use_dynamic_decay == ema_callback.use_dynamic_decay
    assert ema_callback.ema.device == ema_callback.device


def test_ema_update_on_batch_end(model, ema_callback):
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


def test_ema_state_saved_to_checkpoint(model, ema_callback):
    trainer = Trainer()
    ema_callback.on_fit_start(trainer, model)

    checkpoint = {}
    ema_callback.on_save_checkpoint(trainer, model, checkpoint)

    assert "state_dict" in checkpoint or "state_dict_ema" in checkpoint


def test_load_from_checkpoint(model, ema_callback):
    trainer = Trainer()
    ema_callback.on_fit_start(trainer, model)

    checkpoint = {"state_dict": deepcopy(model.state_dict())}
    ema_callback.on_load_checkpoint(checkpoint)

    assert (
        ema_callback.ema.state_dict_ema.keys()
        == checkpoint["state_dict"].keys()
    )


def test_validation_epoch_start_and_end(model, ema_callback):
    trainer = Trainer()
    ema_callback.on_fit_start(trainer, model)

    ema_callback.on_validation_epoch_start(trainer, model)
    assert ema_callback.collected_state_dict is not None

    ema_callback.on_validation_end(trainer, model)
    for k in ema_callback.collected_state_dict.keys():
        assert torch.equal(
            ema_callback.collected_state_dict[k], model.state_dict()[k]
        )
