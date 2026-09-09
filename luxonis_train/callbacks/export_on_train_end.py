"""Exports the model to ONNX when training ends."""

import lightning.pytorch as pl
from loguru import logger

import luxonis_train as lxt
from luxonis_train.registry import CALLBACKS

from .needs_checkpoint import NeedsCheckpoint


@CALLBACKS.register()
class ExportOnTrainEnd(NeedsCheckpoint):
    """Export the model to ONNX when training ends.

    The callback only writes the ONNX file. Use `ConvertOnTrainEnd
    <luxonis_train.callbacks.ConvertOnTrainEnd>` to also archive the
    model and convert it for a device.

    """

    def on_train_end(
        self, _: pl.Trainer, pl_module: "lxt.LuxonisLightningModule"
    ) -> None:
        """Export the model on train end.

        Args:
            _ (``pl.Trainer``): Pytorch Lightning trainer. Unused.
            pl_module (``pl.LightningModule``): Pytorch Lightning module.

        """
        checkpoint = self.get_checkpoint(pl_module)
        if checkpoint is None:  # pragma: no cover
            logger.warning("Skipping model export.")
            return

        pl_module.core.export(weights=checkpoint)
