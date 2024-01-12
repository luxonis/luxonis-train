import logging

import lightning.pytorch as pl
from luxonis_ml.utils.filesystem import LuxonisFileSystem

from luxonis_train.utils.registry import CALLBACKS


@CALLBACKS.register_module()
class UploadCheckpointOnTrainEnd(pl.Callback):
    """Callback that uploads best checkpoint based on the validation loss."""

    def __init__(self, upload_directory: str):
        """Constructs `UploadCheckpointOnTrainEnd`.

        @type upload_directory: str
        @param upload_directory: Path used as upload directory
        """
        super().__init__()
        self.fs = LuxonisFileSystem(
            upload_directory, allow_active_mlflow_run=True, allow_local=False
        )

    def on_train_end(self, trainer: pl.Trainer, _: pl.LightningModule) -> None:
        logger = logging.getLogger(__name__)
        logger.info(f"Started checkpoint upload to {self.fs.full_path()}...")
        model_checkpoint_callbacks = [
            c
            for c in trainer.callbacks  # type: ignore
            if isinstance(c, pl.callbacks.ModelCheckpoint)  # type: ignore
        ]
        # NOTE: assume that first checkpoint callback is based on val loss
        local_path = model_checkpoint_callbacks[0].best_model_path
        self.fs.put_file(
            local_path=local_path,
            remote_path=local_path.split("/")[-1],
            mlflow_instance=trainer.logger.experiment.get(  # type: ignore
                "mlflow", None
            ),
        )
        logger.info("Checkpoint upload finished")
