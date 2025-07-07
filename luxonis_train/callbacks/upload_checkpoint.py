from pathlib import Path
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
        self.last_best_checkpoints = set()

    def on_save_checkpoint(
        self,
        trainer: pl.Trainer,
        module: "lxt.LuxonisLightningModule",
        checkpoint: dict[str, Any],
    ) -> None:
        checkpoint_paths = [
            c.best_model_path
            for c in trainer.checkpoint_callbacks
            if isinstance(c, ModelCheckpoint) and c.best_model_path
        ]
        for curr_best_checkpoint in checkpoint_paths:
            if curr_best_checkpoint not in self.last_best_checkpoints:
                logger.info("Uploading checkpoint...")
                temp_filename = (
                    Path(curr_best_checkpoint).parent.with_suffix(".ckpt").name
                )
                torch.save(  # nosemgrep
                    checkpoint, temp_filename
                )
                module.logger.upload_artifact(temp_filename, typ="weights")

                Path(temp_filename).unlink(missing_ok=True)

                logger.info("Checkpoint upload finished")
                self.last_best_checkpoints.add(curr_best_checkpoint)
