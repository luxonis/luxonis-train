from abc import ABC, abstractmethod

import lightning.pytorch as pl
from luxonis_ml.utils.registry import AutoRegisterMeta
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from luxonis_train.utils.registry import STRATEGIES


class BaseTrainingStrategy(
    ABC,
    metaclass=AutoRegisterMeta,
    register=False,
    registry=STRATEGIES,
):
    def __init__(self, pl_module: pl.LightningModule):
        self.pl_module = pl_module

    @abstractmethod
    def configure_optimizers(
        self,
    ) -> tuple[list[Optimizer], list[LRScheduler]]: ...

    @abstractmethod
    def update_parameters(self, *args, **kwargs) -> None: ...
