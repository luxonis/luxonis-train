from abc import ABC, abstractmethod
from collections.abc import Sequence

from lightning.pytorch.utilities.types import (
    LRSchedulerConfig,
    LRSchedulerTypeUnion,
)
from luxonis_ml.utils.registry import AutoRegisterMeta
from torch.optim import Optimizer

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
    ) -> tuple[
        Sequence[Optimizer],
        Sequence[LRSchedulerTypeUnion | LRSchedulerConfig],
    ]: ...

    @abstractmethod
    def update_parameters(self) -> None: ...
