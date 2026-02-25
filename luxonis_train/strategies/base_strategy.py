from abc import ABC, abstractmethod

from lightning.pytorch.utilities.types import OptimizerLRScheduler
from luxonis_ml.utils.registry import AutoRegisterMeta

import luxonis_train as lxt
from luxonis_train.registry import STRATEGIES


class BaseTrainingStrategy(
    ABC, metaclass=AutoRegisterMeta, register=False, registry=STRATEGIES
):
    @abstractmethod
    def __init__(self, pl_module: "lxt.LuxonisLightningModule", **kwargs): ...

    @abstractmethod
    def configure_optimizers(self) -> OptimizerLRScheduler: ...

    @abstractmethod
    def update_parameters(self) -> None: ...
