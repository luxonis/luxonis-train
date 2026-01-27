import lightning.pytorch as pl
from loguru import logger

import luxonis_train as lxt

from .needs_checkpoint import NeedsCheckpoint


class ConvertOnTrainEnd(NeedsCheckpoint):
    """Callback that exports, archives, and converts the model on train
    end."""

    def on_train_end(
        self, trainer: pl.Trainer, pl_module: "lxt.LuxonisLightningModule"
    ) -> None:
        """Converts the model on train end.

        @type trainer: L{pl.Trainer}
        @param trainer: Pytorch Lightning trainer.
        @type pl_module: L{pl.LightningModule}
        @param pl_module: Pytorch Lightning module.
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
