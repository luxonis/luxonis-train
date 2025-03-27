import lightning.pytorch as pl
from loguru import logger

import luxonis_train as lxt
from luxonis_train.registry import CALLBACKS

from .needs_checkpoint import NeedsCheckpoint


@CALLBACKS.register()
class ExportOnTrainEnd(NeedsCheckpoint):
    def on_train_end(
        self, _: pl.Trainer, pl_module: "lxt.LuxonisLightningModule"
    ) -> None:
        """Exports the model on train end.

        @type trainer: L{pl.Trainer}
        @param trainer: Pytorch Lightning trainer.
        @type pl_module: L{pl.LightningModule}
        @param pl_module: Pytorch Lightning module.
        """
        checkpoint = self.get_checkpoint(pl_module)
        if checkpoint is None:  # pragma: no cover
            logger.warning("Skipping model export.")
            return

        pl_module.core.export(weights=checkpoint)
