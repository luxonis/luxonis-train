import os
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import torch
from loguru import logger

import luxonis_train
from luxonis_train.utils.registry import CALLBACKS


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
        module: "luxonis_train.models.LuxonisLightningModule",
        checkpoint: dict[str, Any],
    ) -> None:
        # Log only once per epoch in case there are multiple ModelCheckpoint callbacks
        if not self.last_logged_epoch == trainer.current_epoch:
            checkpoint_paths = [
                c.best_model_path
                for c in trainer.callbacks  # type: ignore
                if isinstance(c, pl.callbacks.ModelCheckpoint)  # type: ignore
                and c.best_model_path
            ]
            for curr_best_checkpoint in checkpoint_paths:
                if curr_best_checkpoint not in self.last_best_checkpoints:
                    logger.info("Uploading checkpoint...")
                    temp_filename = (
                        Path(curr_best_checkpoint)
                        .parent.with_suffix(".ckpt")
                        .name
                    )
                    torch.save(  # nosemgrep
                        checkpoint, temp_filename
                    )
                    module.logger.upload_artifact(temp_filename, typ="weights")

                    os.remove(temp_filename)

                    logger.info("Checkpoint upload finished")
                    self.last_best_checkpoints.add(curr_best_checkpoint)

            self.last_logged_epoch = trainer.current_epoch
