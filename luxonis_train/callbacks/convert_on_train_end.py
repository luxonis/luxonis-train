"""Exports, archives, and converts the model when training ends."""

import lightning.pytorch as pl
from loguru import logger

import luxonis_train as lxt

from .needs_checkpoint import NeedsCheckpoint


class ConvertOnTrainEnd(NeedsCheckpoint):
    """Export, archive, and convert the model when training ends.

    The callback runs these steps in order:

    1. Export the model to ONNX.
    2. Build an NN Archive around it.
    3. Run ``blobconverter`` when ``exporter.blobconverter.active`` is
       true.
    4. Run the HubAI SDK conversion when ``exporter.hubai.active`` is
       true.

    Prefer this callback over a separate `ExportOnTrainEnd
    <luxonis_train.callbacks.ExportOnTrainEnd>` and `ArchiveOnTrainEnd
    <luxonis_train.callbacks.ArchiveOnTrainEnd>`, which together do the
    first two steps only.

    """

    def on_train_end(
        self, trainer: pl.Trainer, pl_module: "lxt.LuxonisLightningModule"
    ) -> None:
        """Convert the model on train end.

        Args:
            trainer (``pl.Trainer``): Pytorch Lightning trainer.
            pl_module (``pl.LightningModule``): Pytorch Lightning module.

        """
        checkpoint = self.get_checkpoint(pl_module)
        if checkpoint is None:  # pragma: no cover
            logger.warning("Skipping model conversion.")
            return

        # Avoid multiple display error with conversion progress bar
        progress_bar = trainer.progress_bar_callback
        if progress_bar is not None:
            stop_progress = getattr(progress_bar, "_stop_progress", None)
            if callable(stop_progress):
                stop_progress()

        pl_module.core.convert(weights=checkpoint)
