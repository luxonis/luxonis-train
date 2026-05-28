from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING

from lightning.pytorch.utilities.types import (
    LRSchedulerConfig,
    LRSchedulerTypeUnion,
)
from luxonis_ml.utils.registry import AutoRegisterMeta
from torch.optim import Optimizer

import luxonis_train as lxt
from luxonis_train.config.config import OptimizerConfig, SchedulerConfig
from luxonis_train.registry import STRATEGIES

if TYPE_CHECKING:
    from luxonis_train.lightning.optimization_planner import ParameterGroupSpec


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

    @abstractmethod
    def get_base_configs(self) -> tuple[OptimizerConfig, SchedulerConfig]: ...

    def transform_groups(
        self, groups: Sequence["ParameterGroupSpec"]
    ) -> list["ParameterGroupSpec"]:
        return list(groups)

    def create_scheduler(
        self,
        optimizer: Optimizer,
        group_specs: Sequence["ParameterGroupSpec"],
        default_scheduler: LRSchedulerTypeUnion | LRSchedulerConfig,
    ) -> LRSchedulerTypeUnion | LRSchedulerConfig:
        _ = optimizer, group_specs
        return default_scheduler

    def supports_optimizer_override(self, optimizer_name: str) -> bool:
        _ = optimizer_name
        return False

    def supports_multi_optimizer(self) -> bool:
        return False

    def requires_manual_optimization(self, optimizer_count: int) -> bool:
        return optimizer_count > 1
