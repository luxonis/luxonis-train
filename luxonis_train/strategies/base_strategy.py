"""The base class every training strategy inherits."""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from luxonis_ml.utils import AutoRegisterMeta
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
    """Base class for the training strategies.

    A strategy adds parameter-group rules to the partition of the model
    parameters, and can change its groups on every step. A subclass
    registers in the ``STRATEGIES`` registry under its class name, so
    ``trainer.training_strategy.name`` can name it.

    `resolve_training_plan` evaluates the strategy rules after the
    ``finetuning`` entries of every node and before the default rule.
    A node entry therefore wins over a strategy rule, and every
    parameter that the strategy does not claim still gets an optimizer.

    Example:
        The ``trainer`` section of a config that uses a strategy:

        .. code-block:: yaml

            trainer:
              training_strategy:
                name: TripleLRSGDStrategy
                params:
                  lr: 0.02
                  warmup_epochs: 3

    """

    @abstractmethod
    def __init__(self, pl_module: "lxt.LuxonisLightningModule", **kwargs):
        """Create the strategy for a Lightning module.

        `luxonis_train.lightning.utils.build_training_strategy` calls
        the constructor with the module and with the entries of
        ``trainer.training_strategy.params`` as keyword arguments.

        Args:
            pl_module (LuxonisLightningModule): The module to train.
            **kwargs (``Any``): The parameters of the strategy from the
                config.

        """
        ...

    @abstractmethod
    def rules(self) -> list[StrategyRule]:
        """Return the parameter-group rules of the strategy, in order.

        The first rule whose selector accepts a free parameter claims
        it. Rules with the same optimizer name and the same scheduler
        share one inner optimizer.

        Returns:
            list[StrategyRule]: The rules. A rule without a scheduler
            uses the scheduler of `get_base_configs`.

        """
        ...

    @abstractmethod
    def get_base_configs(self) -> tuple[OptimizerConfig, SchedulerConfig]:
        """Return the base optimizer and scheduler of the strategy.

        While the strategy is active, the base configs replace
        ``trainer.optimizer`` and ``trainer.scheduler``. The
        ``finetuning`` entries of the nodes merge their overrides into
        them, and the default rule uses them. An implementation can
        raise ``NotImplementedError``. `resolve_training_plan` then uses
        ``trainer.optimizer`` and ``trainer.scheduler``.

        Returns:
            tuple[OptimizerConfig, SchedulerConfig]: The base optimizer
            config and the base scheduler config.

        """
        ...

    def attach(
        self,
        runtime: TrainingPlanRuntime,
        handles: Mapping[str, tuple[GroupHandle, ...]],
    ) -> None:
        """Store the runtime and the group handles of the strategy
        rules.

        `LuxonisLightningModule.configure_optimizers` calls the method
        after it builds the optimizers. The method sets the ``runtime``
        and ``group_handles`` attributes. An override must keep them, or
        call this method. A handle holds indices, so it stays valid
        after a checkpoint loads.

        Args:
            runtime (TrainingPlanRuntime): The optimizers and schedulers
                of the plan. ``runtime.group(handle)`` returns a group.
            handles (``Mapping[str, tuple[GroupHandle, ...]]``): The
                handles of the groups of each rule, keyed by the rule
                tag. A rule that claims no parameter has no entry.

        """
        self.runtime = runtime
        self.group_handles = handles

    def update_parameters(self) -> None:
        """Change the groups of the strategy after a backward pass.

        `TrainingManager` calls the method after each backward pass,
        before the optimizers step. The base implementation does
        nothing.

        """
        return

    def opaque_parameter_ids(self) -> set[int]:
        """Return the parameters that the strategy claims outside the
        rules.

        `resolve_training_plan` leaves these parameters out of the
        plan. The base implementation returns an empty set. Only
        `LegacyStrategyAdapter` overrides the method.

        Returns:
            set[int]: The ``id()`` of each claimed parameter.

        """
        return set()

    def opaque_inners(self) -> list[tuple[Optimizer, Any]]:
        """Return the optimizers that the strategy builds itself.

        `build_training_plan` adds them after the inner optimizers of
        the plan. The base implementation returns an empty list. Only
        `LegacyStrategyAdapter` overrides the method.

        Returns:
            ``list[tuple[Optimizer, Any]]``: Pairs of an optimizer and
            its scheduler. The scheduler is a scheduler, a Lightning
            scheduler config dictionary, or ``None``.

        """
        return []
