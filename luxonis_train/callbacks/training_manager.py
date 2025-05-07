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
        for (
            node_name,
            node,
            e,
            lr_after_unfreeze,
        ) in pl_module.nodes.frozen_nodes():
            if e == epoch:
                logger.info(f"Unfreezing node '{node_name}'")
                self.unfreeze_and_add_param_group(
                    node, optimizer, lr_after_unfreeze, initial_denom_lr=1.0
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
