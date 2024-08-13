import logging
import os
from pathlib import Path
from typing import cast

import lightning.pytorch as pl

import luxonis_train
from luxonis_train.utils.config import Config
from luxonis_train.utils.registry import CALLBACKS
from luxonis_train.utils.tracker import LuxonisTrackerPL


@CALLBACKS.register_module()
class ArchiveOnTrainEnd(pl.Callback):
    def __init__(self, upload_to_mlflow: bool = False):
        """Callback that performs archiving of onnx or exported model at the end of
        training/export. TODO: description.

        @type upload_to_mlflow: bool
        @param upload_to_mlflow: If set to True, overrides the upload url in Archiver
            with currently active MLFlow run (if present).
        """
        super().__init__()
        self.upload_to_mlflow = upload_to_mlflow

    def on_train_end(
        self,
        trainer: pl.Trainer,
        pl_module: "luxonis_train.models.LuxonisLightningModule",
    ) -> None:
        """Archives the model on train end.

        @type trainer: L{pl.Trainer}
        @param trainer: Pytorch Lightning trainer.
        @type pl_module: L{pl.LightningModule}
        @param pl_module: Pytorch Lightning module.
        @raises RuntimeError: If no best model path is found.
        """
        from luxonis_train.core.archiver import Archiver

        best_model_path = pl_module.core.get_min_loss_checkpoint_path()
        if not best_model_path:
            raise RuntimeError(
                "No best model path found. "
                "Please make sure that ModelCheckpoint callback is present "
                "and at least one validation epoch has been performed."
            )
        cfg: Config = pl_module.cfg
        cfg.model.weights = best_model_path
        if self.upload_to_mlflow:
            if cfg.tracker.is_mlflow:
                tracker = cast(LuxonisTrackerPL, trainer.logger)
                new_upload_url = f"mlflow://{tracker.project_id}/{tracker.run_id}"
                cfg.archiver.upload_url = new_upload_url
            else:
                logging.getLogger(__name__).warning(
                    "`upload_to_mlflow` is set to True, "
                    "but there is  no MLFlow active run, skipping."
                )

        onnx_path = str(Path(best_model_path).parent.with_suffix(".onnx"))
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(
                "Model executable not found. Make sure to run exporter callback before archiver callback"
            )

        archiver = Archiver(cfg=cfg)

        archiver.archive(onnx_path)
