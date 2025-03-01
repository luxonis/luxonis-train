import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint

import luxonis_train as lxt
from luxonis_train.utils.registry import CALLBACKS


@CALLBACKS.register()
class TestOnTrainEnd(pl.Callback):
    """Callback to perform a test run at the end of the training."""

    def on_train_end(
        self,
        trainer: pl.Trainer,
        pl_module: "lxt.LuxonisLightningModule",
    ) -> None:
        # `trainer.test` would delete the paths so we need to save them
        best_paths = {
            hash(callback.monitor): callback.best_model_path
            for callback in trainer.checkpoint_callbacks
            if isinstance(callback, ModelCheckpoint)
        }

        trainer.test(pl_module, pl_module.core.pytorch_loaders["test"])

        # Restore the paths
        for callback in trainer.checkpoint_callbacks:
            if (
                isinstance(callback, ModelCheckpoint)
                and hash(callback.monitor) in best_paths
            ):
                callback.best_model_path = best_paths[hash(callback.monitor)]
