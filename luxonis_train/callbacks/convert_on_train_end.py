"""Exports, archives, and converts the model when training ends."""

import lightning.pytorch as pl
from loguru import logger

import luxonis_train as lxt

from .needs_checkpoint import NeedsCheckpoint


class ConvertOnTrainEnd(NeedsCheckpoint):
    """Export, archive, and convert the model when training ends.

    The callback passes the best checkpoint to `LuxonisModel.convert`,
    which runs these steps in order:

    1. Export the model to ONNX.
    2. Build an NN Archive around it.
    3. Run ``blobconverter`` when ``exporter.blobconverter.active`` is
       true.
    4. Run the HubAI SDK conversion when ``exporter.hubai.active`` is
       true.

    The ``preferred_checkpoint`` parameter of `NeedsCheckpoint` selects
    the best main metric or the lowest validation loss.

    Prefer this callback over a separate `ExportOnTrainEnd
    <luxonis_train.callbacks.ExportOnTrainEnd>` and `ArchiveOnTrainEnd
    <luxonis_train.callbacks.ArchiveOnTrainEnd>`, which together do the
    first two steps only. When ``trainer.callbacks`` lists an active
    instance of this callback, the config deactivates those two.

    When ``trainer.smart_cfg_auto_populate`` is set, the config adds
    this callback to ``trainer.callbacks`` if it is missing. The config
    adds it after the deactivation step, so the added instance leaves
    the other two active.

    """

    def on_train_end(
        self, trainer: pl.Trainer, pl_module: "lxt.LuxonisLightningModule"
    ) -> None:
        """Export, archive, and convert the best checkpoint.

        Lightning calls this hook once when ``trainer.fit`` ends. The
        hook selects a checkpoint with `NeedsCheckpoint.get_checkpoint`.
        When no checkpoint exists, it logs a warning and stops.

        Otherwise the hook calls the ``_stop_progress`` method of the
        progress bar of ``trainer``, when the bar has one. For a rich
        progress bar, the call stops its live display, because the
        conversion shows its own progress bar. The hook then passes the
        checkpoint to `LuxonisModel.convert`. It does not catch the
        errors of that method.

        The conversion loads the checkpoint into
        ``pl_module.core.lightning_module``, which is ``pl_module`` in a
        `LuxonisModel.train` run. It does not restore the earlier
        weights, so that module holds the weights of the checkpoint
        after the hook.

        Args:
            trainer (``pl.Trainer``): The trainer. The hook reads its
                progress bar callback.
            pl_module (LuxonisLightningModule): The model to convert.

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
