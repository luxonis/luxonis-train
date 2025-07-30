from typing import Literal

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint

import luxonis_train as lxt
from luxonis_train.registry import CALLBACKS
from luxonis_train.typing import View


@CALLBACKS.register()
class TestOnTrainEnd(pl.Callback):
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
        # `trainer.test` would delete the paths so we need to save them
        best_paths = {
            hash(callback.monitor): callback.best_model_path
            for callback in trainer.checkpoint_callbacks
            if isinstance(callback, ModelCheckpoint)
        }

        device_before = pl_module.device

        trainer.test(pl_module, pl_module.core.pytorch_loaders[self.view])

        # .test() moves pl_module to "cpu", we move it back to original device after
        pl_module.to(device_before)

        # Restore the paths
        for callback in trainer.checkpoint_callbacks:
            if (
                isinstance(callback, ModelCheckpoint)
                and hash(callback.monitor) in best_paths
            ):
                callback.best_model_path = best_paths[hash(callback.monitor)]
