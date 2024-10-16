import logging
import math
from copy import deepcopy
from typing import Any

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn

logger = logging.getLogger(__name__)


class ModelEma(nn.Module):
    """Model Exponential Moving Average.

    Keeps a moving average of everything in the model.state_dict
    (parameters and buffers).
    """

    def __init__(
        self,
        model: pl.LightningModule,
        decay: float = 0.9999,
        use_dynamic_decay: bool = True,
        decay_tau: float = 2000,
        device: str | None = None,
    ):
        """Constructs `ModelEma`.

        @type model: L{pl.LightningModule}
        @param model: Pytorch Lightning module.
        @type decay: float
        @param decay: Decay rate for the moving average.
        @type use_dynamic_decay: bool
        @param use_dynamic_decay: Use dynamic decay rate.
        @type decay_tau: float
        @param decay_tau: Decay tau for the moving average.
        @type device: str | None
        @param device: Device to perform EMA on.
        """
        super(ModelEma, self).__init__()
        model.eval()
        self.state_dict_ema = deepcopy(model.state_dict())
        model.train()

        for p in self.state_dict_ema.values():
            p.requires_grad = False
        self.updates = 0
        self.decay = decay
        self.use_dynamic_decay = use_dynamic_decay
        self.decay_tau = decay_tau
        self.device = device
        if self.device is not None:
            self.state_dict_ema = {
                k: v.to(device=device) for k, v in self.state_dict_ema.items()
            }

    def update(self, model: pl.LightningModule) -> None:
        """Update the stored parameters using a moving average.

        Source: U{<https://github.com/huggingface/pytorch-image-models/blob/main/timm/utils/model_ema.py>}

        @license: U{Apache License 2.0<https://github.com/huggingface/pytorch-image-models/tree/main?tab=Apache-2.0-1-ov-file#readme>}

        @type model: L{pl.LightningModule}
        @param model: Pytorch Lightning module.
        """
        with torch.no_grad():
            self.updates += 1

            if self.use_dynamic_decay:
                decay = self.decay * (
                    1 - math.exp(-self.updates / self.decay_tau)
                )
            else:
                decay = self.decay

            ema_lerp_values = []
            model_lerp_values = []
            for ema_v, model_v in zip(
                self.state_dict_ema.values(), model.state_dict().values()
            ):
                if ema_v.is_floating_point():
                    if self.device is not None:
                        model_v = model_v.to(device=self.device)
                    ema_lerp_values.append(ema_v)
                    model_lerp_values.append(model_v)
                else:
                    ema_v.copy_(model_v)

            if hasattr(torch, "_foreach_lerp_"):
                torch._foreach_lerp_(
                    ema_lerp_values, model_lerp_values, weight=1.0 - decay
                )
            else:
                torch._foreach_mul_(ema_lerp_values, scalar=decay)
                torch._foreach_add_(
                    ema_lerp_values, model_lerp_values, alpha=1.0 - decay
                )


class EMACallback(Callback):
    """Callback that updates the stored parameters using a moving
    average."""

    def __init__(
        self,
        decay: float = 0.5,
        use_dynamic_decay: bool = True,
        decay_tau: float = 2000,
        device: str | None = None,
    ):
        """Constructs `EMACallback`.

        @type decay: float
        @param decay: Decay rate for the moving average.
        @type use_dynamic_decay: bool
        @param use_dynamic_decay: Use dynamic decay rate. If True, the
            decay rate will be updated based on the number of updates.
        @type decay_tau: float
        @param decay_tau: Decay tau for the moving average.
        @type device: str | None
        @param device: Device to perform EMA on.
        """
        self.decay = decay
        self.use_dynamic_decay = use_dynamic_decay
        self.decay_tau = decay_tau
        self.device = device

        self.ema = None
        self.collected_state_dict = None

    def on_fit_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Initialize `ModelEma` to keep a copy of the moving average of
        the weights.

        @type trainer: L{pl.Trainer}
        @param trainer: Pytorch Lightning trainer.
        @type pl_module: L{pl.LightningModule}
        @param pl_module: Pytorch Lightning module.
        """

        self.ema = ModelEma(
            pl_module,
            decay=self.decay,
            use_dynamic_decay=self.use_dynamic_decay,
            decay_tau=self.decay_tau,
            device=self.device,
        )

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Update the stored parameters using a moving average.

        @type trainer: L{pl.Trainer}
        @param trainer: Pytorch Lightning trainer.
        @type pl_module: L{pl.LightningModule}
        @param pl_module: Pytorch Lightning module.
        @type outputs: Any
        @param outputs: Outputs from the training step.
        @type batch: Any
        @param batch: Batch data.
        @type batch_idx: int
        @param batch_idx: Batch index.
        """
        if batch_idx % trainer.accumulate_grad_batches == 0:
            if self.ema is not None:
                self.ema.update(pl_module)

    def on_validation_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Do validation using the stored parameters. Save the original
        parameters before replacing with EMA version.

        @type trainer: L{pl.Trainer}
        @param trainer: Pytorch Lightning trainer.
        @type pl_module: L{pl.LightningModule}
        @param pl_module: Pytorch Lightning module.
        """

        self.collected_state_dict = deepcopy(pl_module.state_dict())

        if self.ema is not None:
            pl_module.load_state_dict(self.ema.state_dict_ema)

    def on_validation_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Restore original parameters to resume training later.

        @type trainer: L{pl.Trainer}
        @param trainer: Pytorch Lightning trainer.
        @type pl_module: L{pl.LightningModule}
        @param pl_module: Pytorch Lightning module.
        """
        if self.collected_state_dict is not None:
            pl_module.load_state_dict(self.collected_state_dict)

    def on_train_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Update the LightningModule with the EMA weights.

        @type trainer: L{pl.Trainer}
        @param trainer: Pytorch Lightning trainer.
        @type pl_module: L{pl.LightningModule}
        @param pl_module: Pytorch Lightning module.
        """
        if self.ema is not None:
            pl_module.load_state_dict(self.ema.state_dict_ema)
            logger.info("Model weights replaced with the EMA weights.")

    def on_save_checkpoint(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        checkpoint: dict,
    ) -> None:  # or dict?
        """Save the EMA state_dict to the checkpoint.

        @type trainer: L{pl.Trainer}
        @param trainer: Pytorch Lightning trainer.
        @type pl_module: L{pl.LightningModule}
        @param pl_module: Pytorch Lightning module.
        @type checkpoint: dict
        @param checkpoint: Pytorch Lightning checkpoint.
        """
        if self.ema is not None:
            checkpoint["state_dict"] = self.ema.state_dict_ema

    def on_load_checkpoint(self, callback_state: dict) -> None:
        """Load the EMA state_dict from the checkpoint.

        @type callback_state: dict
        @param callback_state: Pytorch Lightning callback state.
        """
        if self.ema is not None:
            self.ema.state_dict_ema = callback_state["state_dict"]
