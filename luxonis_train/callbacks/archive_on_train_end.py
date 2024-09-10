import logging

import lightning.pytorch as pl

import luxonis_train
from luxonis_train.utils.registry import CALLBACKS

from .needs_checkpoint import NeedsCheckpoint

logger = logging.getLogger(__name__)


@CALLBACKS.register_module()
class ArchiveOnTrainEnd(NeedsCheckpoint):
    def on_train_end(
        self,
        _: pl.Trainer,
        pl_module: "luxonis_train.models.LuxonisLightningModule",
    ) -> None:
        """Archives the model on train end.

        @type trainer: L{pl.Trainer}
        @param trainer: Pytorch Lightning trainer.
        @type pl_module: L{pl.LightningModule}
        @param pl_module: Pytorch Lightning module.
        """

        path = self.get_checkpoint(pl_module)
        if path is None:  # pragma: no cover
            logger.warning("Skipping model archiving.")
            return

        onnx_path = pl_module.core._exported_models.get("onnx")
        if onnx_path is None:  # pragma: no cover
            logger.error(
                "Model executable not found. "
                "Make sure to run exporter callback before archiver callback. "
                "Skipping model archiving."
            )
            return

        pl_module.core.archive(onnx_path)
