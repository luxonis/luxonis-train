import lightning.pytorch as pl
from typing_extensions import override

import luxonis_train as lxt


class TrainingManager(pl.Callback):
    """Drives the declarative freeze schedule and the per-step training-
    strategy hook.

    The freeze schedule is applied idempotently at ``setup`` time and at
    the start of every training epoch, so it needs no checkpointed state
    of its own: a resumed run converges to the scheduled state from the
    restored epoch number, while group membership (static), group
    learning rates (optimizer state dict) and scheduler state all round
    trip through Lightning's regular checkpointing.
    """

    @override
    def setup(
        self,
        trainer: pl.Trainer,
        pl_module: "lxt.LuxonisLightningModule",
        stage: str,
    ) -> None:
        _ = trainer
        if stage != "fit":
            return
        # `trainer.current_epoch` is still 0 here even when resuming
        # (the fit loop is restored later, in `restore_training_state`);
        # the first `on_train_epoch_start` converges to the real epoch.
        pl_module.nodes.freeze_schedule.apply(epoch=0)

    @override
    def on_train_epoch_start(
        self, trainer: pl.Trainer, pl_module: "lxt.LuxonisLightningModule"
    ) -> None:
        pl_module.nodes.freeze_schedule.apply(
            epoch=trainer.current_epoch,
            runtime=pl_module.training_plan,
        )

    @override
    def on_after_backward(
        self, trainer: pl.Trainer, pl_module: "lxt.LuxonisLightningModule"
    ) -> None:
        """PyTorch Lightning hook that is called after the backward
        pass.

        @type trainer: pl.Trainer
        @param trainer: The trainer object.
        @type pl_module: pl.LightningModule
        @param pl_module: The pl_module object.
        """
        if pl_module.training_strategy is not None:
            pl_module.training_strategy.update_parameters()
