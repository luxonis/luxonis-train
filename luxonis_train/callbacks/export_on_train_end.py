import logging

import lightning.pytorch as pl

import luxonis_train
from luxonis_train.utils.registry import CALLBACKS

logger = logging.getLogger(__name__)


@CALLBACKS.register_module()
class ExportOnTrainEnd(pl.Callback):
    def on_train_end(
        self,
        _: pl.Trainer,
        pl_module: "luxonis_train.models.LuxonisLightningModule",
    ) -> None:
        """Exports the model on train end.

        @type trainer: L{pl.Trainer}
        @param trainer: Pytorch Lightning trainer.
        @type pl_module: L{pl.LightningModule}
        @param pl_module: Pytorch Lightning module.
        @raises RuntimeError: If no best model path is found.
        """

        best_model_path = pl_module.core.get_best_metric_checkpoint_path()
        if not best_model_path:
            logger.error(
                "No model checkpoint found. "
                "Make sure that `ModelCheckpoint` callback is present "
                "and at least one validation epoch has been performed. "
                "Skipping model export."
            )
            return

        pl_module.core.export(weights=best_model_path)
