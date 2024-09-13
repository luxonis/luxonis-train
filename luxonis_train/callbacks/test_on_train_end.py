import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint

import luxonis_train
from luxonis_train.utils.registry import CALLBACKS


@CALLBACKS.register_module()
class TestOnTrainEnd(pl.Callback):
    """Callback to perform a test run at the end of the training."""

    def on_train_end(
        self,
        trainer: pl.Trainer,
        pl_module: "luxonis_train.models.LuxonisLightningModule",
    ) -> None:
        # `trainer.test` would delete the paths so we need to save them
        best_paths = {
            hash(callback.monitor): callback.best_model_path
            for callback in trainer.callbacks  # type: ignore
            if isinstance(callback, ModelCheckpoint)
        }

        trainer.test(pl_module, pl_module.core.pytorch_loaders["test"])

        # Restore the paths
        for callback in trainer.callbacks:  # type: ignore
            if isinstance(callback, ModelCheckpoint):
                if hash(callback.monitor) in best_paths:
                    callback.best_model_path = best_paths[
                        hash(callback.monitor)
                    ]
