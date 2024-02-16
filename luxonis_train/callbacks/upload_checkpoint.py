import logging
import os
from typing import Any

import lightning.pytorch as pl
import torch
from luxonis_ml.utils.filesystem import LuxonisFileSystem

from luxonis_train.utils.registry import CALLBACKS


@CALLBACKS.register_module()
class UploadCheckpoint(pl.Callback):
    """Callback that uploads best checkpoint based on the validation loss."""

    def __init__(self, upload_directory: str):
        """Constructs `UploadCheckpoint`.

        @type upload_directory: str
        @param upload_directory: Path used as upload directory
        """
        super().__init__()
        self.fs = LuxonisFileSystem(
            upload_directory, allow_active_mlflow_run=True, allow_local=False
        )
        self.logger = logging.getLogger(__name__)
        self.last_logged_epoch = None
        self.last_best_checkpoint = None

    def on_save_checkpoint(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        checkpoint: dict[str, Any],
    ) -> None:
        # Log only once per epoch in case there are multiple ModelCheckpoint callbacks
        if not self.last_logged_epoch == trainer.current_epoch:
            model_checkpoint_callbacks = [
                c
                for c in trainer.callbacks  # type: ignore
                if isinstance(c, pl.callbacks.ModelCheckpoint)  # type: ignore
            ]
            # NOTE: assume that first checkpoint callback is based on val loss
            curr_best_checkpoint = model_checkpoint_callbacks[0].best_model_path

            if self.last_best_checkpoint != curr_best_checkpoint:
                self.logger.info(f"Started checkpoint upload to {self.fs.full_path}...")
                temp_filename = "curr_best_val_loss.ckpt"
                torch.save(checkpoint, temp_filename)
                self.fs.put_file(
                    local_path=temp_filename,
                    remote_path=temp_filename,
                    mlflow_instance=trainer.logger.experiment.get(  # type: ignore
                        "mlflow", None
                    ),
                )
                os.remove(temp_filename)
                self.logger.info("Checkpoint upload finished")
                self.last_best_checkpoint = curr_best_checkpoint

            self.last_logged_epoch = trainer.current_epoch
