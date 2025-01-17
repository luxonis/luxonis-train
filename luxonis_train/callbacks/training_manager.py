import lightning.pytorch as pl

from luxonis_train.strategies.base_strategy import BaseTrainingStrategy


class TrainingManager(pl.Callback):
    def __init__(self, strategy: BaseTrainingStrategy | None = None):
        """Training manager callback that updates the parameters of the
        training strategy.

        @type strategy: BaseTrainingStrategy
        @param strategy: The strategy to be used.
        """
        self.strategy = strategy

    def on_after_backward(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """PyTorch Lightning hook that is called after the backward
        pass.

        @type trainer: pl.Trainer
        @param trainer: The trainer object.
        @type pl_module: pl.LightningModule
        @param pl_module: The pl_module object.
        """
        if self.strategy is not None:
            self.strategy.update_parameters(pl_module)
