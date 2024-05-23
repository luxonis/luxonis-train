import lightning.pytorch as pl

import luxonis_train
from luxonis_train.utils.registry import CALLBACKS


@CALLBACKS.register_module()
class TestOnTrainEnd(pl.Callback):
    """Callback to perform a test run at the end of the training."""

    def on_train_end(
        self, trainer: pl.Trainer, pl_module: "luxonis_train.models.LuxonisModel"
    ) -> None:
        trainer.test(pl_module, pl_module._core.pytorch_loaders["test"])
