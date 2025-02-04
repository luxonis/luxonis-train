import lightning.pytorch as pl
from lightning.pytorch.callbacks import BaseFinetuning
from loguru import logger
from torch import nn
from torch.optim.optimizer import Optimizer


class ModuleFreezer(BaseFinetuning):
    def __init__(self, frozen_modules: list[tuple[nn.Module, int]]):
        """Callback that freezes parts of the model.

        @type frozen_modules: list[tuple[nn.Module, int]]
        @param frozen_modules: List of tuples of modules and epochs to
            freeze until.
        """
        super().__init__()
        self.frozen_modules = frozen_modules

    def freeze_before_training(self, _: pl.LightningModule) -> None:
        for module, _e in self.frozen_modules:
            logger.info(f"Freezing module {module.__class__.__name__}")
            self.freeze(module, train_bn=False)

    def finetune_function(
        self, pl_module: pl.LightningModule, epoch: int, optimizer: Optimizer
    ) -> None:
        for module, e in self.frozen_modules:
            if epoch == e:
                logger.info(f"Unfreezing module {module.__class__.__name__}")
                self.unfreeze_and_add_param_group(module, optimizer)
