import lightning.pytorch as pl
from lightning.pytorch.callbacks import BaseFinetuning
from loguru import logger
from torch.optim.optimizer import Optimizer
from typing_extensions import override

import luxonis_train as lxt


class TrainingManager(BaseFinetuning):
    @override
    def freeze_before_training(
        self, pl_module: "lxt.LuxonisLightningModule"
    ) -> None:
        for node_name, node, _, _ in pl_module.nodes.frozen_nodes():
            logger.info(f"Freezing node '{node_name}'")
            self.freeze(node, train_bn=False)

    @override
    def finetune_function(
        self,
        pl_module: "lxt.LuxonisLightningModule",
        epoch: int,
        optimizer: Optimizer,
    ) -> None:
        _ = optimizer
        for (
            node_name,
            node,
            e,
            lr_after_unfreeze,
        ) in pl_module.nodes.frozen_nodes():
            # `e <= epoch` rather than `e == epoch`: `freeze_before_training`
            # re-freezes on every `setup`, so a run resumed past the
            # unfreeze epoch would otherwise never unfreeze the node
            # again. The `requires_grad` guard keeps this a no-op once
            # the node is already trainable.
            if e <= epoch and any(
                not parameter.requires_grad for parameter in node.parameters()
            ):
                logger.info(f"Unfreezing node '{node_name}'")
                self.make_trainable(node)
                pl_module.nodes.restore_unfrozen_parameters(
                    node, lr_after_unfreeze
                )

    @override
    def on_train_epoch_start(
        self, trainer: pl.Trainer, pl_module: "lxt.LuxonisLightningModule"
    ) -> None:
        super().on_train_epoch_start(trainer, pl_module)

        # `BaseFinetuning` checkpoints `_internal_optimizer_metadata` and
        # restores `optimizer.param_groups` from it when training is
        # resumed. It only records groups *appended* to the optimizer it
        # is currently iterating over, but `finetune_function` mutates
        # every optimizer at once and may extend an existing group in
        # place. Re-snapshot the real state so a resumed run does not
        # restore stale parameter groups.
        mapping = {
            parameter: name for name, parameter in pl_module.named_parameters()
        }
        for optimizer_index, optimizer in enumerate(trainer.optimizers):
            self._internal_optimizer_metadata[optimizer_index] = (
                self._apply_mapping_to_param_groups(
                    optimizer.param_groups, mapping
                )
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
