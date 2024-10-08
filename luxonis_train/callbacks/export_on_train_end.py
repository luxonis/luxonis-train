import logging

import lightning.pytorch as pl

import luxonis_train
from luxonis_train.utils.registry import CALLBACKS

from .needs_checkpoint import NeedsCheckpoint

logger = logging.getLogger(__name__)


@CALLBACKS.register_module()
class ExportOnTrainEnd(NeedsCheckpoint):
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
        """
        checkpoint = self.get_checkpoint(pl_module)
        if checkpoint is None:  # pragma: no cover
            logger.warning("Skipping model export.")
            return

        pl_module.core.export(weights=checkpoint)
