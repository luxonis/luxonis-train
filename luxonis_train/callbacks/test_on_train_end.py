import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint

import luxonis_train as lxt
from luxonis_train.registry import CALLBACKS


@CALLBACKS.register()
class TestOnTrainEnd(pl.Callback):
    """Callback to perform a test run at the end of the training."""

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

        trainer.test(pl_module, pl_module.core.pytorch_loaders["test"])

        # .test() moves pl_module to "cpu", we move it back to original device after
        pl_module.to(device_before)

        # Restore the paths
        for callback in trainer.checkpoint_callbacks:
            if (
                isinstance(callback, ModelCheckpoint)
                and hash(callback.monitor) in best_paths
            ):
                callback.best_model_path = best_paths[hash(callback.monitor)]
