"""Exports the model to ONNX when training ends."""

import lightning.pytorch as pl
from loguru import logger

import luxonis_train as lxt
from luxonis_train.registry import CALLBACKS

from .needs_checkpoint import NeedsCheckpoint


@CALLBACKS.register()
class ExportOnTrainEnd(NeedsCheckpoint):
    """Export the model to ONNX when training ends.

    The callback exports the best checkpoint. The
    ``preferred_checkpoint`` parameter of `NeedsCheckpoint` selects the
    best main metric or the lowest validation loss.

    The callback does not build an NN Archive and does not convert the
    model for a device. Use `ConvertOnTrainEnd
    <luxonis_train.callbacks.ConvertOnTrainEnd>` for all three steps.
    When ``trainer.callbacks`` lists an active `ConvertOnTrainEnd`, the
    config deactivates this callback. A `ConvertOnTrainEnd` that
    ``trainer.smart_cfg_auto_populate`` adds does not deactivate it.

    """

    def on_train_end(
        self, _: pl.Trainer, pl_module: "lxt.LuxonisLightningModule"
    ) -> None:
        """Export the best checkpoint to ONNX.

        Lightning calls this hook once when ``trainer.fit`` ends. The
        hook selects a checkpoint with `NeedsCheckpoint.get_checkpoint`
        and passes it to `LuxonisModel.export`. That method writes
        ``<name>.onnx`` to ``<run_save_dir>/export``, where ``<name>``
        is ``exporter.name`` or ``model.name``. For a model with one
        input, it also writes the ``modelconverter`` config
        ``<name>.yaml``. It uploads these files to the run when
        ``exporter.upload_to_run`` is set. When no checkpoint exists,
        the hook logs a warning and exports nothing.

        The export loads the checkpoint into ``pl_module`` only for the
        export. After the hook, ``pl_module`` holds its earlier weights
        again.

        Args:
            _ (``pl.Trainer``): The trainer. Unused.
            pl_module (LuxonisLightningModule): The model to export.

        """
        checkpoint = self.get_checkpoint(pl_module)
        if checkpoint is None:  # pragma: no cover
            logger.warning("Skipping model export.")
            return

        pl_module.core.export(weights=checkpoint)
