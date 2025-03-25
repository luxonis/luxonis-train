import tempfile
from typing import Any

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from loguru import logger

import luxonis_train as lxt
from luxonis_train.registry import CALLBACKS


@CALLBACKS.register()
class UploadCheckpoint(pl.Callback):
    """Callback that uploads best checkpoint based on the validation
    loss."""

    def __init__(self):
        """Constructs `UploadCheckpoint`.

        @type upload_directory: str
        @param upload_directory: Path used as upload directory
        """
        super().__init__()
        self.last_logged_epoch = None
        self.last_best_checkpoints = set()

    def on_save_checkpoint(
        self,
        trainer: pl.Trainer,
        module: "lxt.LuxonisLightningModule",
        checkpoint: dict[str, Any],
    ) -> None:
        # Log only once per epoch in case there are multiple ModelCheckpoint callbacks
        if self.last_logged_epoch != trainer.current_epoch:
            checkpoint_paths = [
                c.best_model_path
                for c in trainer.checkpoint_callbacks
                if isinstance(c, ModelCheckpoint) and c.best_model_path
            ]
            for curr_best_checkpoint in checkpoint_paths:
                if curr_best_checkpoint not in self.last_best_checkpoints:
                    logger.info("Uploading checkpoint...")
                    with tempfile.NamedTemporaryFile() as temp_file:
                        torch.save(checkpoint, temp_file.name)  # nosemgrep
                        module.tracker.upload_artifact(
                            temp_file.name, typ="weights"
                        )

                    logger.info("Checkpoint upload finished")
                    self.last_best_checkpoints.add(curr_best_checkpoint)

            self.last_logged_epoch = trainer.current_epoch
