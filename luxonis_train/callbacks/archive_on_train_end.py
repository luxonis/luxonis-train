"""Creates an NN Archive when training ends."""

import lightning.pytorch as pl
from loguru import logger

import luxonis_train as lxt
from luxonis_train.registry import CALLBACKS

from .needs_checkpoint import NeedsCheckpoint


@CALLBACKS.register()
class ArchiveOnTrainEnd(NeedsCheckpoint):
    """Create an NN Archive when training ends.

    The callback archives the ONNX file of an earlier export, for
    example the file of an `ExportOnTrainEnd` that comes before it in
    ``trainer.callbacks``. Without such a file, it exports the best
    checkpoint first. The ``preferred_checkpoint`` parameter of
    `NeedsCheckpoint` selects the best main metric or the lowest
    validation loss.

    `ConvertOnTrainEnd` also builds the archive. When
    ``trainer.callbacks`` lists an active `ConvertOnTrainEnd`, the
    config deactivates this callback. A `ConvertOnTrainEnd` that
    ``trainer.smart_cfg_auto_populate`` adds does not deactivate it.

    """

    def on_train_end(
        self, _: pl.Trainer, pl_module: "lxt.LuxonisLightningModule"
    ) -> None:
        """Build an NN Archive from the ONNX file of the model.

        Lightning calls this hook once when ``trainer.fit`` ends. The
        hook takes the last ONNX file that `LuxonisModel.export` wrote
        for ``pl_module.core``. When no ONNX export ran, the hook selects
        a checkpoint with `NeedsCheckpoint.get_checkpoint` and exports it
        to ONNX first. It then passes the ONNX file to
        `LuxonisModel.archive`, which writes the archive to
        ``<run_save_dir>/archive``. That method also uploads the archive
        to the run when ``archiver.upload_to_run`` is set.

        When no ONNX export ran and no checkpoint exists, the hook logs a
        warning and builds no archive. It logs an error and builds no
        archive when its own export produced no ONNX file.

        The export and the archive do not restore the earlier weights of
        ``pl_module``. After an export of the hook, ``pl_module`` holds
        the checkpoint weights. `LuxonisModel.archive` then loads the
        weights of the `LuxonisModel` constructor, or ``model.weights``
        of the config, when either exists.

        Args:
            _ (``pl.Trainer``): The trainer. Unused.
            pl_module (LuxonisLightningModule): The model to archive.

        """
        onnx_path = pl_module.core._exported_models.get("onnx")
        if onnx_path is None:  # pragma: no cover
            checkpoint = self.get_checkpoint(pl_module)
            if checkpoint is None:
                logger.warning("Skipping model archiving.")
                return
            logger.info("Exported model not found. Exporting to ONNX...")
            pl_module.core.export(weights=checkpoint)
            onnx_path = pl_module.core._exported_models.get("onnx")

        if onnx_path is None:  # pragma: no cover
            logger.error(
                "Model executable not found and couldn't be created. "
                "Skipping model archiving."
            )
            return

        pl_module.core.archive(onnx_path)
