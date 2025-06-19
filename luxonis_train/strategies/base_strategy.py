from abc import ABC, abstractmethod
from typing import Any

from luxonis_ml.utils.registry import AutoRegisterMeta
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

import luxonis_train as lxt
from luxonis_train.registry import STRATEGIES


class BaseTrainingStrategy(
    ABC, metaclass=AutoRegisterMeta, register=False, registry=STRATEGIES
):
    @abstractmethod
    def __init__(self, pl_module: "lxt.LuxonisLightningModule", **kwargs): ...

    @abstractmethod
    def configure_optimizers(
        self,
    ) -> tuple[list[Optimizer], list[LRScheduler | dict[str, Any]]]: ...

    @abstractmethod
    def update_parameters(self) -> None: ...
