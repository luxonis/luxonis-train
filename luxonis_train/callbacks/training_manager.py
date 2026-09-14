"""Applies the node freeze schedule and the training strategy hook."""

import lightning.pytorch as pl
from typing_extensions import override

import luxonis_train as lxt


class TrainingManager(pl.Callback):
    """Callback that applies the freeze schedule and the strategy hook.

    `Nodes.build_callbacks` adds this callback to every run, before the
    callbacks of the config. The `CALLBACKS` registry also holds it. An
    entry in ``trainer.callbacks`` adds a second instance, and each hook
    then runs twice.

    The ``freezing`` section of each node config defines the freeze
    schedule. A node with ``freezing.active`` stays frozen before its
    unfreeze epoch, and trains from that epoch on.
    `FreezeSchedule.apply` derives ``requires_grad`` and the batch
    normalization state from the epoch number and the original values of
    the node. A repeated call for the same epoch gives the same state.
    Thus the callback keeps no state in the checkpoint:

    - A resumed run gets the scheduled state at the start of its first
      epoch.
    - A frozen parameter stays in its parameter group, so the groups
      never change.
    - The optimizer and scheduler state dicts of the Lightning
      checkpoint restore the learning rates of the groups.

    """

    @override
    def setup(
        self,
        trainer: pl.Trainer,
        pl_module: "lxt.LuxonisLightningModule",
        stage: str,
    ) -> None:
        """Apply the freeze schedule for epoch 0 when a fit starts.

        Lightning calls this hook at the start of every ``fit``,
        ``validate``, ``test``, or ``predict`` call. The hook acts only
        for ``fit``. It calls `FreezeSchedule.apply` with epoch 0 and
        without the training plan, so it changes no learning rate. For
        each node whose unfreeze epoch is above 0, the call turns off
        ``requires_grad`` of the parameters. It also turns off
        ``track_running_stats`` of the batch normalization layers of
        that node. It logs a message for each node that it freezes.

        The hook always uses epoch 0. On a resumed run,
        ``trainer.current_epoch`` is still 0 here, because Lightning
        restores the loop state later. `on_train_epoch_start` then
        applies the real epoch.

        Args:
            trainer (``pl.Trainer``): The trainer. Unused.
            pl_module (LuxonisLightningModule): The module that holds the
                freeze schedule in ``nodes.freeze_schedule``.
            stage (str): The stage that starts: ``"fit"``,
                ``"validate"``, ``"test"``, or ``"predict"``.

        """
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
        """Apply the freeze schedule for the epoch that starts.

        Lightning calls this hook at the start of every training epoch.
        The hook calls `FreezeSchedule.apply` with
        ``trainer.current_epoch`` and
        `LuxonisLightningModule.training_plan`. A node before its
        unfreeze epoch stays frozen. A node that reaches its unfreeze
        epoch gets back its original ``requires_grad`` and
        ``track_running_stats`` values. When the node config sets
        ``freezing.lr_after_unfreeze``, the call uses that value on the
        exact unfreeze epoch only. It sets the value as the base learning
        rate of each parameter group of the node. The call logs a message
        each time a node freezes or unfreezes.

        Args:
            trainer (``pl.Trainer``): The trainer. The hook reads its
                ``current_epoch``.
            pl_module (LuxonisLightningModule): The module that holds the
                freeze schedule and the training plan.

        """
        pl_module.nodes.freeze_schedule.apply(
            epoch=trainer.current_epoch,
            runtime=pl_module.training_plan,
        )

    @override
    def on_after_backward(
        self, trainer: pl.Trainer, pl_module: "lxt.LuxonisLightningModule"
    ) -> None:
        """Call the per-step hook of the training strategy.

        Lightning calls this hook after each backward pass, before the
        optimizer updates the parameters. With gradient accumulation,
        several backward passes come before one optimizer step. When
        ``pl_module.training_strategy`` is set, the hook calls its
        `BaseTrainingStrategy.update_parameters`. For example,
        `TripleLRSGDStrategy` sets the warmup learning rates of its
        parameter groups there. Without a strategy, the hook does
        nothing.

        Args:
            trainer (``pl.Trainer``): The trainer. Unused.
            pl_module (LuxonisLightningModule): The module that holds the
                training strategy.

        """
        if pl_module.training_strategy is not None:
            pl_module.training_strategy.update_parameters()
