import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from loguru import logger

import luxonis_train as lxt
from luxonis_train.registry import CALLBACKS
from luxonis_train.typing import View

from .needs_checkpoint import NeedsCheckpoint


@CALLBACKS.register()
class TestOnTrainEnd(NeedsCheckpoint):
    def __init__(self, view: View = "test") -> None:
        """Callback to perform a test run at the end of the training.

        @type view: Literal["train", "val", "test"]
        @param view: The view to use for testing. Defaults to "test".
        """
        super().__init__()
        self.view: View = view

    def on_train_end(
        self, trainer: pl.Trainer, pl_module: "lxt.LuxonisLightningModule"
    ) -> None:
        checkpoint = self.get_checkpoint(pl_module)
        if checkpoint is None:  # pragma: no cover
            logger.warning(
                "Best model checkpoint not found. Using last checkpoint for testing."
            )
        # `trainer.test` would delete the paths so we need to save them
        best_paths = {
            hash(callback.monitor): callback.best_model_path
            for callback in trainer.checkpoint_callbacks
            if isinstance(callback, ModelCheckpoint)
        }

        device_before = pl_module.device

        pl_module.core.test(weights=checkpoint, view=self.view)

        # .test() moves pl_module to "cpu", we move it back to original device after
        pl_module.to(device_before)

        # Restore the paths
        for callback in trainer.checkpoint_callbacks:
            if (
                isinstance(callback, ModelCheckpoint)
                and hash(callback.monitor) in best_paths
            ):
                callback.best_model_path = best_paths[hash(callback.monitor)]
