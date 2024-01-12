import logging
from pathlib import Path
from typing import cast

import lightning.pytorch as pl

from luxonis_train.utils.config import Config
from luxonis_train.utils.registry import CALLBACKS
from luxonis_train.utils.tracker import LuxonisTrackerPL


@CALLBACKS.register_module()
class ExportOnTrainEnd(pl.Callback):
    def __init__(self, upload_to_mlflow: bool = False):
        """Callback that performs export on train end with best weights according to the
        validation loss.

        @type upload_to_mlflow: bool
        @param upload_to_mlflow: If set to True, overrides the upload url in Exporter
            with currently active MLFlow run (if present).
        """
        super().__init__()
        self.upload_to_mlflow = upload_to_mlflow

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Exports the model on train end.

        @type trainer: L{pl.Trainer}
        @param trainer: Pytorch Lightning trainer.
        @type pl_module: L{pl.LightningModule}
        @param pl_module: Pytorch Lightning module.
        @raises RuntimeError: If no best model path is found.
        """
        from luxonis_train.core.exporter import Exporter

        model_checkpoint_callbacks = [
            c
            for c in trainer.callbacks  # type: ignore
            if isinstance(c, pl.callbacks.ModelCheckpoint)  # type: ignore
        ]
        # NOTE: assume that first checkpoint callback is based on val loss
        best_model_path = model_checkpoint_callbacks[0].best_model_path
        if not best_model_path:
            raise RuntimeError(
                "No best model path found. "
                "Please make sure that ModelCheckpoint callback is present "
                "and at least one validation epoch has been performed."
            )
        cfg: Config = pl_module.cfg
        cfg.model.weights = best_model_path
        if self.upload_to_mlflow:
            if pl_module.cfg.tracker.is_mlflow:
                tracker = cast(LuxonisTrackerPL, trainer.logger)
                new_upload_directory = f"mlflow://{tracker.project_id}/{tracker.run_id}"
                cfg.exporter.upload_directory = new_upload_directory
            else:
                logging.getLogger(__name__).warning(
                    "`upload_to_mlflow` is set to True, "
                    "but there is  no MLFlow active run, skipping."
                )
        exporter = Exporter(cfg=cfg)
        onnx_path = str(Path(best_model_path).parent.with_suffix(".onnx"))
        exporter.export(onnx_path=onnx_path)
