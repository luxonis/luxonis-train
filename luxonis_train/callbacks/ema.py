import logging
from typing import Any, List, Tuple, Union

import pytorch_lightning as pl
import torch
from copy import deepcopy
from torch import nn
from torch import Tensor
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT
import math


logger = logging.getLogger(__name__)


class ModelEma(nn.Module):
    """Model Exponential Moving Average.
    Keeps a moving average of everything in the model.state_dict (parameters and buffers).
    """

    def __init__(self, model: pl.LightningModule, decay: float = 0.9999, use_dynamic_decay: bool = True, device: str = None):
        """Constructs `ModelEma`.

        @type model: L{pl.LightningModule}
        @param model: Pytorch Lightning module.
        @type decay: float
        @param decay: Decay rate for the moving average.
        @type use_dynamic_decay: bool
        @param use_dynamic_decay: Use dynamic decay rate.
        @type device: str
        @param device: Device to perform EMA on.
        """
        super(ModelEma, self).__init__()
        model.eval()
        self.state_dict = deepcopy(model.state_dict())
        model.train()

        for p in self.state_dict.values():
            p.requires_grad = False
        self.updates = 0
        self.decay = decay
        self.use_dynamic_decay = use_dynamic_decay
        self.device = device
        if self.device is not None:
            self.state_dict = {k: v.to(device=device) for k, v in self.state_dict.items()}

    def update(self, model: pl.LightningModule) -> None:
        """Update the stored parameters using a moving average

        @type model: L{pl.LightningModule}
        @param model: Pytorch Lightning module.
        """
        with torch.no_grad():
            for k, ema_p in self.state_dict.items():
                if ema_p.dtype.is_floating_point:
                    self.updates += 1

                    if self.use_dynamic_decay:
                        decay = self.decay * (1 - math.exp(-self.updates / 2000))
                    else:
                        decay = self.decay

                    model_p = model.state_dict()[k]
                    if self.device is not None:
                        model_p = model_p.to(device=self.device)
                    ema_p *= decay
                    ema_p += (1. - decay) * model_p

class EMACallback(Callback):
    """
    Callback that updates the stored parameters using a moving average.
    """

    def __init__(self, decay: float = 0.9999, use_dynamic_decay: bool = True, use_ema_weights: bool = True, device: str = None):
        """Constructs `EMACallback`.

        @type decay: float
        @param decay: Decay rate for the moving average.
        @type use_dynamic_decay: bool
        @param use_dynamic_decay: Use dynamic decay rate. If True, the decay rate will be updated based on the number of updates.
        @type use_ema_weights: bool
        @param use_ema_weights: Use EMA weights (replace model weights with EMA weights)
        @type device: str
        @param device: Device to perform EMA on.
        """
        self.decay = decay
        self.use_dynamic_decay = use_dynamic_decay
        self.use_ema_weights = use_ema_weights
        self.device = device

        self.ema = None
        self.collected_state_dict = None

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Initialize `ModelEma` to keep a copy of the moving average of the weights

        @type trainer: L{pl.Trainer}
        @param trainer: Pytorch Lightning trainer.
        @type pl_module: L{pl.LightningModule}
        @param pl_module: Pytorch Lightning module.
        """

        self.ema = ModelEma(pl_module, decay=self.decay, use_dynamic_decay = self.use_dynamic_decay, device=self.device)

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Update the stored parameters using a moving average

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
        
        self.ema.update(pl_module)

    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Do validation using the stored parameters. Save the original parameters before replacing with EMA version.

        @type trainer: L{pl.Trainer}
        @param trainer: Pytorch Lightning trainer.
        @type pl_module: L{pl.LightningModule}
        @param pl_module: Pytorch Lightning module.
        """

        self.collected_state_dict = deepcopy(pl_module.state_dict())

        pl_module.load_state_dict(self.ema.state_dict)

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Restore original parameters to resume training later

        @type trainer: L{pl.Trainer}
        @param trainer: Pytorch Lightning trainer.
        @type pl_module: L{pl.LightningModule}
        @param pl_module: Pytorch Lightning module.
        """
        pl_module.load_state_dict(self.collected_state_dict)

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Update the LightningModule with the EMA weights 

        @type trainer: L{pl.Trainer}
        @param trainer: Pytorch Lightning trainer.
        @type pl_module: L{pl.LightningModule}
        @param pl_module: Pytorch Lightning module.
        """
        if self.use_ema_weights:
            pl_module.load_state_dict(self.ema.state_dict)
            logger.info("Model weights replaced with the EMA weights.")

    def on_save_checkpoint(self, trainer: pl.Trainer, pl_module: pl.LightningModule, checkpoint: dict) -> None: # or dict?
        """Save the EMA state_dict to the checkpoint

        @type trainer: L{pl.Trainer}
        @param trainer: Pytorch Lightning trainer.
        @type pl_module: L{pl.LightningModule}
        @param pl_module: Pytorch Lightning module.
        @type checkpoint: dict
        @param checkpoint: Pytorch Lightning checkpoint.
        """
        if self.use_ema_weights:
            checkpoint["state_dict"] = self.ema.state_dict
        elif self.ema is not None:
            checkpoint["state_dict_ema"] = self.ema.state_dict

    def on_load_checkpoint(self, callback_state: dict) -> None:
        """Load the EMA state_dict from the checkpoint
        
        @type callback_state: dict
        @param callback_state: Pytorch Lightning callback state.
        """
        if self.use_ema_weights:
            self.ema.state_dict = callback_state["state_dict"]
        elif self.ema is not None:
            self.ema.state_dict = callback_state["state_dict_ema"]