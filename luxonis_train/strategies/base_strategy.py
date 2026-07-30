from abc import ABC, abstractmethod
from collections.abc import Sequence

from lightning.pytorch.utilities.types import (
    LRSchedulerConfig,
    LRSchedulerTypeUnion,
)
from luxonis_ml.typing import Kwargs
from luxonis_ml.utils.registry import AutoRegisterMeta
from torch import nn
from torch.optim import Optimizer

import luxonis_train as lxt
from luxonis_train.config.config import OptimizerConfig, SchedulerConfig
from luxonis_train.registry import STRATEGIES


class BaseTrainingStrategy(
    ABC, metaclass=AutoRegisterMeta, register=False, registry=STRATEGIES
):
    @abstractmethod
    def __init__(self, pl_module: "lxt.LuxonisLightningModule", **kwargs): ...

    @abstractmethod
    def configure_optimizers(
        self, excluded_params: set[int] | None = None
    ) -> tuple[
        Sequence[Optimizer],
        Sequence[LRSchedulerTypeUnion | LRSchedulerConfig],
    ]: ...

    @abstractmethod
    def update_parameters(self) -> None: ...

    @abstractmethod
    def get_base_configs(self) -> tuple[OptimizerConfig, SchedulerConfig]: ...

    def estimate_optimizer_count(
        self, excluded_params: set[int] | None = None
    ) -> int:
        _ = excluded_params
        return 1

    def attach_optimizers(self, optimizers: Sequence[Optimizer]) -> None:
        """Register optimizers that were built outside of the strategy.

        Parameters claimed by C{finetuning} rules end up in their own
        optimizers, which the strategy does not own. Strategies that
        adjust learning rates during training should override this to
        keep those optimizers in sync with their own.

        @type optimizers: Sequence[Optimizer]
        @param optimizers: The optimizers built from the C{finetuning}
            rules.
        """
        _ = optimizers

    def split_parameter_group(
        self,
        parameters: list[tuple[nn.Module, nn.Parameter]],
        optimizer_params: Kwargs,
        explicit_keys: set[str],
    ) -> list[tuple[list[nn.Parameter], Kwargs]]:
        """Split the parameters matched by one C{finetuning} rule into
        optimizer parameter groups.

        Strategies that apply different optimizer options depending on
        the kind of parameter should override this, so that rules
        inheriting the strategy's base config do not silently get
        options the strategy itself would never apply.

        @type parameters: list[tuple[nn.Module, nn.Parameter]]
        @param parameters: The matched parameters together with the
            module that owns them.
        @type optimizer_params: Kwargs
        @param optimizer_params: The optimizer options of the rule,
            already merged with the strategy's base config.
        @type explicit_keys: set[str]
        @param explicit_keys: Options the user set on the rule itself.
            These must be left alone, as they express explicit intent.
        @rtype: list[tuple[list[nn.Parameter], Kwargs]]
        @return: One entry per parameter group to create.
        """
        _ = explicit_keys
        return [([parameter for _, parameter in parameters], optimizer_params)]
