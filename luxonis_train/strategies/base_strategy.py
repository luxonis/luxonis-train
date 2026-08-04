from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from luxonis_ml.utils.registry import AutoRegisterMeta
from torch.optim import Optimizer

import luxonis_train as lxt
from luxonis_train.config.config import OptimizerConfig, SchedulerConfig
from luxonis_train.lightning.training_plan import (
    GroupHandle,
    StrategyRule,
    TrainingPlanRuntime,
)
from luxonis_train.registry import STRATEGIES


class BaseTrainingStrategy(
    ABC, metaclass=AutoRegisterMeta, register=False, registry=STRATEGIES
):
    """A training strategy contributes parameter-group rules to the
    model-wide partition and may adjust its groups every step.

    Strategy rules are evaluated after every node ``finetuning`` rule
    and before the default tail rule, so node-level overrides take
    precedence and every parameter the strategy does not claim still
    ends up in an optimizer.
    """

    @abstractmethod
    def __init__(self, pl_module: "lxt.LuxonisLightningModule", **kwargs): ...

    @abstractmethod
    def rules(self) -> list[StrategyRule]:
        """Ordered parameter-group rules of the strategy."""
        ...

    @abstractmethod
    def get_base_configs(self) -> tuple[OptimizerConfig, SchedulerConfig]:
        """Return the base optimizer/scheduler pair.

        Used as the inheritance base for node ``finetuning`` rules that
        omit names and as the specification of the default tail rule
        while this strategy is active.
        """
        ...

    def attach(
        self,
        runtime: TrainingPlanRuntime,
        handles: Mapping[str, tuple[GroupHandle, ...]],
    ) -> None:
        """Receive the handles of the groups produced by this strategy's
        rules, keyed by rule tag.

        Called once after the optimizers are built. Handles are index-
        based and stay valid across checkpoint restores.
        """
        self.runtime = runtime
        self.group_handles = handles

    def update_parameters(self) -> None:
        """Per-step hook, called after the backward pass and before the
        optimizers step.
        """
        return

    def opaque_parameter_ids(self) -> set[int]:
        """Parameter ids claimed outside the rule system.

        Only the legacy-strategy adapter overrides this.
        """
        return set()

    def opaque_inners(self) -> list[tuple[Optimizer, Any]]:
        """Pre-built optimizers (with their schedulers) to mount as
        additional inner optimizers.

        Only the legacy-strategy adapter overrides this.
        """
        return []
