"""Uploads each new best checkpoint to the tracker."""

from copy import copy
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from loguru import logger
from typing_extensions import override

import luxonis_train as lxt
from luxonis_train.registry import CALLBACKS

_CHECKPOINT_STATE_KEY = "last_best_checkpoints"


@CALLBACKS.register()
class UploadCheckpoint(pl.Callback):
    """Callback that uploads each new best checkpoint to the tracker.

    The callback reads every ``ModelCheckpoint`` of the trainer. A run
    has one on the lowest validation loss, and one on the main metric
    when the config has a metric. When the best checkpoint of such a
    callback changes, `on_save_checkpoint` uploads it.
    `LuxonisTrackerPL` sends the file to MLFlow and to Weights and
    Biases, when the run uses them.

    When ``trainer.smart_cfg_auto_populate`` is set,
    `Config.smart_auto_populate` adds this callback to
    ``trainer.callbacks`` if it is missing. `LuxonisModel.tune` removes
    it from the config of each trial.

    """

    def __init__(self):
        """Initialize the callback with no uploaded checkpoints."""
        super().__init__()
        self._last_best_checkpoints: set[str] = set()

    @override
    def on_save_checkpoint(
        self,
        trainer: pl.Trainer,
        module: "lxt.LuxonisLightningModule",
        checkpoint: dict[str, Any],
    ) -> None:
        """Upload the best checkpoints that are not uploaded yet.

        Lightning calls this hook each time the trainer saves a full
        checkpoint, before the ``on_save_checkpoint`` hook of the module.
        A save with ``weights_only`` does not call it. The hook makes a
        shallow copy of ``checkpoint``. It adds the run metadata that
        `LuxonisLightningModule.on_save_checkpoint` describes to the
        copy. That step runs a forward pass, which leaves ``module`` in
        evaluation mode.

        The hook then reads the ``best_model_path`` of each
        ``ModelCheckpoint`` of ``trainer``. For each non-empty path that
        the callback did not upload before, the hook does these steps:

        1. It records the path as uploaded. It writes the new callback
           state to ``checkpoint["callbacks"]`` and to the copy.
        2. It writes the copy to ``<directory>.ckpt`` in the current
           working directory. ``<directory>`` is the name of the
           directory that holds the path. For the checkpoints that
           `Nodes.build_callbacks` adds, the names are
           ``min_val_loss.ckpt`` and ``best_val_metric.ckpt``. The write
           replaces a file of that name.
        3. It uploads the file with ``module.logger.upload_artifact`` and
           the artifact type ``weights``. The upload runs on rank zero
           only.
        4. It deletes the file.

        When the write or the upload raises an error, the hook removes
        the path from the record. It writes the callback state without
        that path to ``checkpoint["callbacks"]`` and raises the error
        again.

        The hook logs an info message before and after each upload.

        The uploaded file holds the state of the current save, not the
        file at ``best_model_path``. A ``ModelCheckpoint`` sets its new
        best path just before it saves, so the two hold the same
        weights. Lightning collects the callback states before it calls
        this hook. Thus the hook must write the new state into
        ``checkpoint`` itself. A resumed run restores the uploaded paths
        and does not upload the current state for a historical best
        path.

        Args:
            trainer (``pl.Trainer``): The trainer. The hook reads its
                checkpoint callbacks.
            module (LuxonisLightningModule): The model. The hook uploads
                through its logger and adds its metadata to the copy.
            checkpoint (``dict[str, Any]``): The checkpoint dictionary
                that Lightning is about to write. For each new upload,
                the hook replaces the state of this callback in
                ``checkpoint["callbacks"]``. It does not change the
                other keys.

        """
        upload_checkpoint = copy(checkpoint)
        upload_checkpoint["callbacks"] = copy(checkpoint["callbacks"])
        module._add_custom_data_to_checkpoint(upload_checkpoint)
        checkpoint_paths = [
            c.best_model_path
            for c in trainer.checkpoint_callbacks
            if isinstance(c, ModelCheckpoint) and c.best_model_path
        ]
        for curr_best_checkpoint in checkpoint_paths:
            if curr_best_checkpoint in self._last_best_checkpoints:
                continue

            logger.info("Uploading checkpoint...")
            self._last_best_checkpoints.add(curr_best_checkpoint)
            callback_state = self.state_dict()
            checkpoint["callbacks"][self.state_key] = callback_state
            upload_checkpoint["callbacks"][self.state_key] = callback_state
            temp_filename = (
                Path(curr_best_checkpoint).parent.with_suffix(".ckpt").name
            )
            try:
                torch.save(  # nosemgrep
                    upload_checkpoint, temp_filename
                )
                module.logger.upload_artifact(temp_filename, typ="weights")
            except Exception:
                self._last_best_checkpoints.remove(curr_best_checkpoint)
                checkpoint["callbacks"][self.state_key] = self.state_dict()
                raise

            Path(temp_filename).unlink(missing_ok=True)
            logger.info("Checkpoint upload finished")

    @override
    def state_dict(self) -> dict[str, set[str]]:
        """Return the paths of the checkpoints already uploaded.

        Returns:
            dict[str, set[str]]: The uploaded paths under
                ``"last_best_checkpoints"``.

        """
        return {
            _CHECKPOINT_STATE_KEY: self._last_best_checkpoints.copy(),
        }

    @override
    def load_state_dict(self, state_dict: dict[str, set[str]]) -> None:
        """Restore the paths of the checkpoints already uploaded.

        Args:
            state_dict (dict[str, set[str]]): The callback state from a
                checkpoint. An old checkpoint can hold an empty dictionary.

        """
        self._last_best_checkpoints = state_dict.get(
            _CHECKPOINT_STATE_KEY, set()
        ).copy()
