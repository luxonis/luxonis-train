"""Keeps an exponential moving average of the weights.

Validation during a fit runs with the average weights, so its
visualizations also come from these weights. After the fit, the model
keeps the average weights. A checkpoint that is not weights-only holds
the average weights as the model weights. An export from such a
checkpoint thus uses them too.

"""

import math
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

import lightning.pytorch as pl
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from loguru import logger
from torch import Tensor, nn

from luxonis_train.utils.checkpoint import filter_checkpoint_state_dict


class ModelEma(nn.Module):
    """Exponential moving average of the state dictionary of a model.

    The average covers every entry of ``model.state_dict()``: the
    parameters and the persistent buffers. It lives in a plain
    dictionary, not in registered buffers, so ``ModelEma.state_dict()``
    does not return it. `EMACallback` creates and updates it.

    Attributes:
        state_dict_ema (``dict[str, Tensor]``): The average, with the
            keys of ``model.state_dict()``.
        updates (int): The number of updates of the average. Each
            `update` call adds ``1``. `EMACallback.on_fit_start` sets it
            from a checkpoint that holds a count.
        decay (float): The largest decay of the average.
        use_dynamic_decay (bool): Whether the decay grows with
            ``updates``.
        decay_tau (float): The time constant of the dynamic decay.

    """

    def __init__(
        self,
        model: pl.LightningModule,
        decay: float = 0.9999,
        use_dynamic_decay: bool = True,
        decay_tau: float = 2000,
    ):
        r"""Copy the state dictionary of ``model`` as the first average.

        The method calls ``model.eval()``, makes a deep copy of
        ``model.state_dict()``, and then calls ``model.train()``. Thus
        ``model`` is in training mode afterwards, whatever its mode was
        before. The copy does not require gradients.

        Args:
            model (``pl.LightningModule``): The model to average.
            decay (float): The largest decay :math:`d`. A value near
                ``1`` moves the average slowly.
            use_dynamic_decay (bool): When ``True``, the decay grows from
                ``0`` toward ``decay`` as the updates add up. See
                `update`.
            decay_tau (float): The time constant :math:`\tau` of the
                dynamic decay, in updates. A larger value makes the decay
                grow more slowly.

        """
        super().__init__()
        model.eval()
        self.state_dict_ema = deepcopy(model.state_dict())
        model.train()

        for p in self.state_dict_ema.values():
            p.requires_grad = False
        self.updates = 0
        self._decay = decay
        self._use_dynamic_decay = use_dynamic_decay
        self._decay_tau = decay_tau

    def update(self, model: pl.LightningModule) -> None:
        r"""Move the average one step toward the state of ``model``.

        The method adds ``1`` to ``updates`` and computes the decay
        :math:`d_t` of this step. With ``use_dynamic_decay``, the decay
        grows from ``0`` toward ``decay`` as the updates add up, so the
        first averages follow the model closely:

        .. math::

            d_t = d \left(1 - e^{-t / \tau}\right)

        Here :math:`t` is ``updates``, :math:`d` is ``decay``, and
        :math:`\tau` is ``decay_tau``. Without ``use_dynamic_decay``,
        :math:`d_t = d`. The method then changes each floating point
        entry :math:`\theta_{\text{ema}}` of the average in place:

        .. math::

            \theta_{\text{ema}} \leftarrow d_t \, \theta_{\text{ema}}
            + (1 - d_t) \, \theta

        Here :math:`\theta` is the entry of ``model.state_dict()`` with
        the same key. The method copies each entry of another type,
        such as the ``num_batches_tracked`` counter of a batch norm. It
        skips a key that ``model`` does not have. No gradients flow
        through the update.

        Args:
            model (``pl.LightningModule``): The model whose current state
                the average moves toward.

        References:
            - Source: adapted from `timm model_ema.py
              <https://github.com/huggingface/pytorch-image-models/blob/main/timm/utils/model_ema.py>`_
              (`Apache License 2.0
              <https://github.com/huggingface/pytorch-image-models/tree/main?tab=Apache-2.0-1-ov-file#readme>`_).

        Examples:
            A fixed decay of ``0.5`` moves the average halfway:

            >>> import lightning.pytorch as pl
            >>> import torch
            >>> model = pl.LightningModule()
            >>> model.weight = torch.nn.Parameter(torch.zeros(2))
            >>> ema = ModelEma(model, decay=0.5, use_dynamic_decay=False)
            >>> _ = model.weight.data.fill_(1.0)
            >>> ema.update(model)
            >>> ema.state_dict_ema["weight"].tolist()
            [0.5, 0.5]

            The dynamic decay is smaller for the first updates, so the
            average moves farther toward the model:

            >>> _ = model.weight.data.zero_()
            >>> ema = ModelEma(model, decay=0.5, decay_tau=1.0)
            >>> _ = model.weight.data.fill_(1.0)
            >>> ema.update(model)
            >>> round(ema.state_dict_ema["weight"][0].item(), 3)
            0.684
            >>> ema.updates
            1

        """
        with torch.no_grad():
            self.updates += 1

            if self._use_dynamic_decay:
                decay = self._decay * (
                    1 - math.exp(-self.updates / self._decay_tau)
                )
            else:
                decay = self._decay

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
    """Callback that keeps an exponential moving average of the weights.

    The callback keeps the average in a `ModelEma`. It changes the model
    at these points of a run:

    - `on_fit_start` creates the average from the current weights.
    - `on_train_batch_end` moves the average toward the trained weights
      once in each gradient accumulation window.
    - Validation and test run with the average weights, when the
      average exists. The callback keeps a copy of the model weights
      when the loop starts, and loads the copy back when the loop ends.
    - When training ends, the model keeps the average weights.
    - Each checkpoint that is not weights-only holds the average weights
      as the model weights. The state of the callback in the checkpoint
      also holds the average and the update count, so a resumed fit
      continues the average.

    While `replace_weights` holds explicit weights in the model, the
    callback does not swap any weights.

    `LuxonisLightningModule.configure_callbacks` builds a new instance
    of this callback for each call of the trainer, such as
    ``trainer.fit`` or ``trainer.test``. Thus a ``trainer.test`` call
    after the fit has no average, and the test runs with the current
    weights of the model.

    The config moves this callback to the front of
    ``trainer.callbacks``, so that it runs before the other callbacks of
    the config.

    Attributes:
        decay (float): The largest decay of the average.
        use_dynamic_decay (bool): Whether the decay grows with the
            number of updates.
        decay_tau (float): The time constant of the dynamic decay.
        loaded_ema_state_dict (``Mapping[str, Tensor] | None``): The
            average from a checkpoint, which the next `on_fit_start`
            applies. ``None`` when there is no such average.
        loaded_ema_updates (int | None): The update count from a
            checkpoint, which the next `on_fit_start` applies. ``None``
            when there is no such count.
        collected_state_dict (``dict[str, Tensor] | None``): The copy of
            the model weights that the last weight swap kept. During a
            fit, these are the trained weights. ``None`` before the
            first swap.

    """

    def __init__(
        self,
        decay: float = 0.5,
        use_dynamic_decay: bool = True,
        decay_tau: float = 2000,
    ):
        """Initialize the callback with the options of the average.

        The callback creates the average in `on_fit_start`, not here.

        Args:
            decay (float): The largest decay of the average. A value near
                ``1`` moves the average slowly. The default ``0.5`` is far
                lower than the ``0.9999`` default of `ModelEma`.
            use_dynamic_decay (bool): When ``True``, the decay grows from
                ``0`` toward ``decay`` as the updates add up. See
                `ModelEma.update`.
            decay_tau (float): The time constant of the dynamic decay, in
                updates.

        """
        self._decay = decay
        self._use_dynamic_decay = use_dynamic_decay
        self._decay_tau = decay_tau

        self._ema = None
        self._loaded_ema_state_dict = None
        self._loaded_ema_updates = None
        self._collected_state_dict = None

    @staticmethod
    def _format_key_list(keys: set[str]) -> str:
        return ", ".join(sorted(keys)) if keys else "<none>"

    @property
    def ema(self) -> ModelEma:
        """The `ModelEma` that holds the average.

        Raises:
            ValueError: When `on_fit_start` has not created the average
                yet.

        """
        if self._ema is None:
            raise ValueError("Ema model not yet initialized.")
        return self._ema

    def on_fit_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Create the average from the current weights of the model.

        Lightning calls this hook at the start of ``trainer.fit``. The
        hook builds a new `ModelEma` from ``pl_module`` with the options
        of the callback. When `load_state_dict` or `on_load_checkpoint`
        stored an average before, the hook copies it into the new
        average:

        - It ignores the entries that losses, metrics, and visualizers
          keep for their node, in both averages.
        - It logs a warning for the keys that the stored average misses,
          the keys it has in excess, and the keys with another shape.
        - It keeps the new values for the missing keys and for the keys
          with another shape.
        - It moves the stored tensors to the device of the new average.
        - It sets the update count when a count was stored.

        It then clears the stored average and count.

        Args:
            trainer (``pl.Trainer``): The trainer. Unused.
            pl_module (``pl.LightningModule``): The model to average.
                `ModelEma` leaves it in training mode.

        """
        self._ema = ModelEma(
            pl_module,
            decay=self._decay,
            use_dynamic_decay=self._use_dynamic_decay,
            decay_tau=self._decay_tau,
        )
        if self._loaded_ema_state_dict is None:
            return
        self._restore_loaded_ema_state(self._loaded_ema_state_dict)

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Move the average one step toward the trained weights.

        Lightning calls this hook after each training batch. The hook
        calls `ModelEma.update` when the average exists and
        ``batch_idx`` is a multiple of ``trainer.accumulate_grad_batches``.
        Thus the average moves once in each gradient accumulation
        window, on the first batch of the window.

        Args:
            trainer (``pl.Trainer``): The trainer. The hook reads its
                ``accumulate_grad_batches``.
            pl_module (``pl.LightningModule``): The model whose weights
                the average moves toward.
            outputs (``STEP_OUTPUT``): The output of the training step.
                Unused.
            batch (``Any``): The batch. Unused.
            batch_idx (int): The index of the batch in the epoch.

        """
        if (
            self._ema is not None
            and batch_idx % trainer.accumulate_grad_batches == 0
        ):
            self._ema.update(pl_module)

    def on_validation_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Load the average weights into the model for validation.

        Lightning calls this hook at the start of each validation epoch.
        The hook keeps a deep copy of the state dictionary of
        ``pl_module`` in ``collected_state_dict``. It then loads the
        average into ``pl_module`` when the average exists. While
        `replace_weights` holds explicit weights in the model, the hook
        does nothing.

        Args:
            trainer (``pl.Trainer``): The trainer. Unused.
            pl_module (``pl.LightningModule``): The model to validate.

        """
        self._swap_to_ema_weights(pl_module)

    def on_validation_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Load the kept weights back into the model after validation.

        Lightning calls this hook when the validation loop ends. The hook
        loads ``collected_state_dict`` into ``pl_module``. During a fit,
        these are the trained weights. The hook does nothing when no
        copy exists, or while `replace_weights` holds explicit weights in
        the model. The copy stays in the callback.

        Args:
            trainer (``pl.Trainer``): The trainer. Unused.
            pl_module (``pl.LightningModule``): The validated model.

        """
        self._restore_original_weights(pl_module)

    def on_test_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Load the average weights into the model for the test.

        Lightning calls this hook at the start of each test epoch. The
        hook swaps the weights like `on_validation_epoch_start`: it keeps
        a deep copy of the weights of ``pl_module``, then loads the
        average when it exists. While `replace_weights` holds explicit
        weights in the model, the hook does nothing.

        Args:
            trainer (``pl.Trainer``): The trainer. Unused.
            pl_module (``pl.LightningModule``): The model to test.

        """
        self._swap_to_ema_weights(pl_module)

    def on_test_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Load the kept weights back into the model after the test.

        Lightning calls this hook when the test loop ends. The hook loads
        ``collected_state_dict`` into ``pl_module``, like
        `on_validation_end`. It does nothing when no copy exists, or
        while `replace_weights` holds explicit weights in the model.

        Args:
            trainer (``pl.Trainer``): The trainer. Unused.
            pl_module (``pl.LightningModule``): The tested model.

        """
        self._restore_original_weights(pl_module)

    def on_train_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Load the average weights into the model when training ends.

        Lightning calls this hook once when ``trainer.fit`` ends. The
        hook swaps the weights like `on_validation_epoch_start`, but it
        does not load the trained weights back. Thus the model keeps the
        average weights after the fit. While `replace_weights` holds
        explicit weights in the model, the hook does nothing.

        Args:
            trainer (``pl.Trainer``): The trainer. Unused.
            pl_module (``pl.LightningModule``): The trained model.

        """
        self._swap_to_ema_weights(pl_module)

    def on_save_checkpoint(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        checkpoint: dict,
    ) -> None:
        """Store the average as the model weights of the checkpoint.

        Lightning calls this hook when it saves a checkpoint that is not
        weights-only, after it collects the output of `state_dict`. When
        the average exists, the hook replaces ``checkpoint["state_dict"]``
        with the average. A model that loads the checkpoint thus gets the
        average weights.

        Args:
            trainer (``pl.Trainer``): The trainer. Unused.
            pl_module (``pl.LightningModule``): The model. Unused.
            checkpoint (dict): The checkpoint that Lightning saves. The
                hook changes it in place.

        """
        if self._ema is not None:
            checkpoint["state_dict"] = self._ema.state_dict_ema

    def state_dict(self) -> dict[str, Any]:
        """Return the state of the callback for a checkpoint.

        Lightning calls this method when it saves a checkpoint that is
        not weights-only. It stores a non-empty result in the
        ``callbacks`` entry of the checkpoint, under the state key of
        the callback. It does not store an empty result.

        Returns:
            ``dict[str, Any]``: An empty dictionary when the average does
            not exist yet. Otherwise a dictionary with two keys:

            - ``"ema_state_dict"``: the average, without the entries that
              losses, metrics, and visualizers keep for their node;
            - ``"updates"``: the number of updates of the average.

        """
        if self._ema is None:
            return {}
        return {
            "ema_state_dict": filter_checkpoint_state_dict(
                self._ema.state_dict_ema
            ),
            "updates": self._ema.updates,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Store a checkpoint average for the next `on_fit_start`.

        Lightning calls this method when it restores a checkpoint, with
        the dictionary that `state_dict` returned. The method reads the
        value under ``"ema_state_dict"``, or under ``"state_dict"`` when
        that key is missing. It keeps that value in
        ``loaded_ema_state_dict`` when the value is a mapping. It keeps
        the value under ``"updates"`` in ``loaded_ema_updates`` when
        that value is an ``int``. It ignores a value of another type and
        an empty ``state_dict``. The current average does not change.

        Args:
            state_dict (``dict[str, Any]``): The state of the callback.

        Example:
            >>> import torch
            >>> callback = EMACallback()
            >>> callback.load_state_dict(
            ...     {"ema_state_dict": {"weight": torch.ones(1)}, "updates": 7}
            ... )
            >>> sorted(callback.loaded_ema_state_dict)
            ['weight']
            >>> callback.loaded_ema_updates
            7

        """
        self._load_ema_state(state_dict)

    def on_load_checkpoint(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        callback_state: dict,
    ) -> None:
        """Store the checkpoint weights for the next `on_fit_start`.

        Lightning calls this hook when it restores a checkpoint that has
        a ``callbacks`` key, before it calls `load_state_dict`. Lightning
        passes the whole checkpoint, not the state of the callback. The
        hook reads it like `load_state_dict`. The checkpoint has no
        ``"ema_state_dict"`` key at the top level, so the hook stores its
        ``"state_dict"``: the average weights, when this callback saved
        the checkpoint. `load_state_dict` then replaces that value with
        the ``"ema_state_dict"`` of the callback state, when the
        checkpoint has one.

        Args:
            trainer (``pl.Trainer``): The trainer. Unused.
            pl_module (``pl.LightningModule``): The model. Unused.
            callback_state (dict): The whole checkpoint, despite the
                name.

        """
        self._load_ema_state(callback_state)

    def _restore_loaded_ema_state(
        self, loaded_checkpoint: Mapping[str, Tensor]
    ) -> None:
        loaded_state = filter_checkpoint_state_dict(loaded_checkpoint)
        current_state = self.ema.state_dict_ema
        comparable_current = filter_checkpoint_state_dict(current_state)
        incompatible = self._warn_about_state_differences(
            comparable_current, loaded_state
        )
        target_device = next(iter(current_state.values())).device
        for key, value in loaded_state.items():
            if key in current_state and key not in incompatible:
                current_state[key] = value.to(target_device)
        self.ema.state_dict_ema = current_state
        if self._loaded_ema_updates is not None:
            self.ema.updates = self._loaded_ema_updates
        self._loaded_ema_state_dict = None
        self._loaded_ema_updates = None

    def _warn_about_state_differences(
        self,
        current_state: Mapping[str, torch.Tensor],
        loaded_state: Mapping[str, torch.Tensor],
    ) -> set[str]:
        current_keys, loaded_keys = set(current_state), set(loaded_state)
        missing, extra = current_keys - loaded_keys, loaded_keys - current_keys
        incompatible = {
            key
            for key in current_keys & loaded_keys
            if current_state[key].shape != loaded_state[key].shape
        }
        self._warn_for_state_keys(
            missing,
            "EMA checkpoint is missing keys present in the current model. "
            "Keeping freshly initialized EMA values for: ",
        )
        self._warn_for_state_keys(
            extra,
            "EMA checkpoint contains keys not present in the current model. Ignoring: ",
        )
        self._warn_for_state_keys(
            incompatible,
            "EMA checkpoint contains keys with incompatible shapes. Ignoring: ",
        )
        return incompatible

    def _warn_for_state_keys(self, keys: set[str], message: str) -> None:
        if keys:
            logger.warning(f"{message}{self._format_key_list(keys)}")

    def _load_ema_state(self, state_dict: dict[str, Any]) -> None:
        if state_dict:
            loaded_state_dict = state_dict.get(
                "ema_state_dict", state_dict.get("state_dict")
            )
            if isinstance(loaded_state_dict, Mapping):
                self._loaded_ema_state_dict = loaded_state_dict
            updates = state_dict.get("updates")
            if isinstance(updates, int):
                self._loaded_ema_updates = updates

    def _swap_to_ema_weights(self, pl_module: pl.LightningModule) -> None:
        """Keep a copy of the model weights, then load the average.

        The method keeps a deep copy of the state dictionary in
        ``collected_state_dict``, also when the average does not exist.
        It does nothing while `replace_weights` holds explicit weights
        in the model.

        Args:
            pl_module (``pl.LightningModule``): The model.

        """
        if getattr(pl_module, "_weights_explicitly_loaded", False):
            return
        self._collected_state_dict = deepcopy(pl_module.state_dict())
        if self._ema is not None:
            pl_module.load_state_dict(self._ema.state_dict_ema)

    def _restore_original_weights(self, pl_module: pl.LightningModule) -> None:
        """Load the weights that the last swap kept back into the model.

        The method does nothing when ``collected_state_dict`` is
        ``None``, or while `replace_weights` holds explicit weights in
        the model.

        Args:
            pl_module (``pl.LightningModule``): The model.

        """
        if getattr(pl_module, "_weights_explicitly_loaded", False):
            return
        if self._collected_state_dict is not None:
            pl_module.load_state_dict(self._collected_state_dict)
