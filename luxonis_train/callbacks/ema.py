import math
from copy import deepcopy
from typing import Any

import lightning.pytorch as pl
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from loguru import logger
from torch import nn


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
        """
        super().__init__()
        model.eval()
        self.state_dict_ema = deepcopy(model.state_dict())
        model.train()

        for p in self.state_dict_ema.values():
            p.requires_grad = False
        self.updates = 0
        self.decay = decay
        self.use_dynamic_decay = use_dynamic_decay
        self.decay_tau = decay_tau

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

            model_state_dict = model.state_dict()
            ema_lerp_values = []
            model_lerp_values = []
            for key, ema_v in self.state_dict_ema.items():
                model_v = model_state_dict.get(key)
                if model_v is None:
                    continue
                if ema_v.is_floating_point():
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


class EMACallback(pl.Callback):
    """Callback that updates the stored parameters using a moving
    average."""

    def __init__(
        self,
        decay: float = 0.5,
        use_dynamic_decay: bool = True,
        decay_tau: float = 2000,
    ):
        """Constructs `EMACallback`.

        @type decay: float
        @param decay: Decay rate for the moving average.
        @type use_dynamic_decay: bool
        @param use_dynamic_decay: Use dynamic decay rate. If True, the
            decay rate will be updated based on the number of updates.
        @type decay_tau: float
        @param decay_tau: Decay tau for the moving average.
        """
        self.decay = decay
        self.use_dynamic_decay = use_dynamic_decay
        self.decay_tau = decay_tau

        self._ema = None
        self.loaded_ema_state_dict = None
        self.loaded_ema_updates = None
        self.collected_state_dict = None

    @staticmethod
    def _format_key_list(keys: set[str]) -> str:
        return ", ".join(sorted(keys)) if keys else "<none>"

    @property
    def ema(self) -> ModelEma:
        if self._ema is None:
            raise ValueError("Ema model not yet initalized.")
        return self._ema

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
        self._ema = ModelEma(
            pl_module,
            decay=self.decay,
            use_dynamic_decay=self.use_dynamic_decay,
            decay_tau=self.decay_tau,
        )
        if self.loaded_ema_state_dict is not None:
            target_device = next(
                iter(self._ema.state_dict_ema.values())
            ).device
            current_state_dict = self._ema.state_dict_ema
            current_keys = set(current_state_dict)
            loaded_keys = set(self.loaded_ema_state_dict)
            missing_in_checkpoint = current_keys - loaded_keys
            extra_in_checkpoint = loaded_keys - current_keys
            incompatible_shapes = {
                key
                for key in current_keys & loaded_keys
                if current_state_dict[key].shape
                != self.loaded_ema_state_dict[key].shape
            }

            if missing_in_checkpoint:
                logger.warning(
                    "EMA checkpoint is missing keys present in the current model. "
                    "Keeping freshly initialized EMA values for: "
                    f"{self._format_key_list(missing_in_checkpoint)}"
                )
            if extra_in_checkpoint:
                logger.warning(
                    "EMA checkpoint contains keys not present in the current model. "
                    "Ignoring: "
                    f"{self._format_key_list(extra_in_checkpoint)}"
                )
            if incompatible_shapes:
                logger.warning(
                    "EMA checkpoint contains keys with incompatible shapes. "
                    "Ignoring: "
                    f"{self._format_key_list(incompatible_shapes)}"
                )

            for key, value in self.loaded_ema_state_dict.items():
                if (
                    key in current_state_dict
                    and key not in incompatible_shapes
                ):
                    current_state_dict[key] = value.to(target_device)
            self._ema.state_dict_ema = current_state_dict
            if self.loaded_ema_updates is not None:
                self._ema.updates = self.loaded_ema_updates
            self.loaded_ema_state_dict = None
            self.loaded_ema_updates = None

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
        if (
            self._ema is not None
            and batch_idx % trainer.accumulate_grad_batches == 0
        ):
            self._ema.update(pl_module)

    def on_validation_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Swap the model's weights to the EMA weights at the start of
        validation.

        @type trainer: L{pl.Trainer}
        @param trainer: Pytorch Lightning trainer.
        @type pl_module: L{pl.LightningModule}
        @param pl_module: Pytorch Lightning module.
        """
        self._swap_to_ema_weights(pl_module)

    def on_validation_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Restore the original model weights after validation.

        @type trainer: L{pl.Trainer}
        @param trainer: Pytorch Lightning trainer.
        @type pl_module: L{pl.LightningModule}
        @param pl_module: Pytorch Lightning module.
        """
        self._restore_original_weights(pl_module)

    def on_test_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Swap the model's weights to the EMA weights at the start of
        testing.

        @type trainer: L{pl.Trainer}
        @param trainer: Pytorch Lightning trainer.
        @type pl_module: L{pl.LightningModule}
        @param pl_module: Pytorch Lightning module.
        """
        self._swap_to_ema_weights(pl_module)

    def on_test_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Restore the original model weights after testing.

        @type trainer: L{pl.Trainer}
        @param trainer: Pytorch Lightning trainer.
        @type pl_module: L{pl.LightningModule}
        @param pl_module: Pytorch Lightning module.
        """
        self._restore_original_weights(pl_module)

    def on_train_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Replace the model's weights with the EMA weights at the end
        of training.

        This final update ensures that the trained model uses the EMA
        weights.
        @type trainer: L{pl.Trainer}
        @param trainer: Pytorch Lightning trainer.
        @type pl_module: L{pl.LightningModule}
        @param pl_module: Pytorch Lightning module.
        """
        self._swap_to_ema_weights(pl_module)

    def on_save_checkpoint(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        checkpoint: dict,
    ) -> dict[str, Any] | None:
        """Save the EMA state dictionary into the checkpoint.

        @type trainer: L{pl.Trainer}
        @param trainer: Pytorch Lightning trainer.
        @type pl_module: L{pl.LightningModule}
        @param pl_module: Pytorch Lightning module.
        @type checkpoint: dict
        @param checkpoint: Pytorch Lightning checkpoint.
        """
        if self._ema is not None:
            checkpoint["state_dict"] = self._ema.state_dict_ema
            return {
                "ema_state_dict": self._ema.state_dict_ema,
                "updates": self._ema.updates,
            }
        return None

    def on_load_checkpoint(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        callback_state: dict,
    ) -> None:
        """Load the EMA state dictionary from the checkpoint.

        @type callback_state: dict
        @param callback_state: Pytorch Lightning callback state.
        """
        if callback_state:
            self.loaded_ema_state_dict = callback_state.get(
                "ema_state_dict", callback_state.get("state_dict")
            )
            updates = callback_state.get("updates")
            if isinstance(updates, int):
                self.loaded_ema_updates = updates

    def _swap_to_ema_weights(self, pl_module: pl.LightningModule) -> None:
        """Swap the current model weights with the EMA weights.

        The current state is saved so that it can be restored later.
        """
        if getattr(pl_module, "_weights_explicitly_loaded", False):
            return
        self.collected_state_dict = deepcopy(pl_module.state_dict())
        if self._ema is not None:
            pl_module.load_state_dict(self._ema.state_dict_ema)

    def _restore_original_weights(self, pl_module: pl.LightningModule) -> None:
        """Restore the model's original weights.

        This method reverts the model to its state prior to the EMA
        weight swap.
        """
        if getattr(pl_module, "_weights_explicitly_loaded", False):
            return
        if self.collected_state_dict is not None:
            pl_module.load_state_dict(self.collected_state_dict)
