"""The adapter for a strategy of the previous ``configure_optimizers``
API.
"""

from typing import Any

from torch.optim import Optimizer
from typing_extensions import override

from luxonis_train.config.config import OptimizerConfig, SchedulerConfig
from luxonis_train.lightning.training_plan import StrategyRule

from .base_strategy import BaseTrainingStrategy

DEPRECATION_MESSAGE = (
    "Training strategy '{name}' implements the deprecated strategy API "
    "(`configure_optimizers`/`update_parameters`). It has been mounted "
    "in compatibility mode and will keep working until the next minor "
    "release. Port it to the new API: implement `rules()` returning "
    "parameter-group rules and (optionally) `update_parameters()` using "
    "the group handles passed to `attach()`."
)


class LegacyStrategyAdapter(BaseTrainingStrategy, register=False):
    """An adapter for a strategy of the deprecated
    ``configure_optimizers`` API.

    `luxonis_train.lightning.utils.build_training_strategy` logs a
    deprecation warning and wraps such a strategy in this adapter. The
    optimizers and schedulers that ``configure_optimizers`` of the
    legacy strategy returns become extra inner optimizers of the plan.

    The parameters of these optimizers stay out of the rule partition.
    The node ``finetuning`` entries and the default rule claim every
    parameter that the legacy optimizers leave out, frozen parameters
    included. Such a parameter therefore trains after its node
    unfreezes.

    """

    def __init__(self, legacy: Any):
        """Wrap a legacy strategy.

        The constructor does not call ``configure_optimizers``. The first
        call of `opaque_parameter_ids` or `opaque_inners` calls it once,
        and the adapter keeps the result.

        Args:
            legacy (``Any``): The legacy strategy. It must have the
                methods ``configure_optimizers`` and
                ``update_parameters``, and can have
                ``get_base_configs``. ``configure_optimizers`` returns a
                sequence of optimizers and a sequence of schedulers.

        """
        self._legacy = legacy
        self._mounted: tuple[list[Optimizer], list[Any]] | None = None

    @property
    def legacy_name(self) -> str:
        """The class name of the legacy strategy.

        `resolve_training_plan` names the strategy with it when the
        legacy optimizers hold a parameter that a ``finetuning`` entry
        claims.

        """
        return type(self._legacy).__name__

    def _mount(self) -> tuple[list[Optimizer], list[Any]]:
        if self._mounted is None:
            optimizers, schedulers = self._legacy.configure_optimizers()
            optimizers = list(optimizers)
            schedulers = list(schedulers)
            if len(schedulers) < len(optimizers):
                schedulers += [None] * (len(optimizers) - len(schedulers))
            self._mounted = (optimizers, schedulers)
        return self._mounted

    @override
    def rules(self) -> list[StrategyRule]:
        """Return no rules, because the legacy strategy has none.

        Returns:
            list[StrategyRule]: An empty list.

        """
        return []

    @override
    def get_base_configs(self) -> tuple[OptimizerConfig, SchedulerConfig]:
        """Return the base configs of the legacy strategy.

        `luxonis_train.lightning.utils.build_training_strategy` gives a
        legacy class without ``get_base_configs`` a stub that raises
        ``NotImplementedError``. When this method raises that error,
        `resolve_training_plan` uses ``trainer.optimizer`` and
        ``trainer.scheduler``.

        Returns:
            tuple[OptimizerConfig, SchedulerConfig]: The result of
            ``get_base_configs()`` of the legacy strategy.

        Raises:
            NotImplementedError: If the legacy strategy has no
                ``get_base_configs`` attribute.

        """
        get_base_configs = getattr(self._legacy, "get_base_configs", None)
        if get_base_configs is None:
            raise NotImplementedError
        return get_base_configs()

    @override
    def update_parameters(self) -> None:
        """Call ``update_parameters()`` of the legacy strategy."""
        self._legacy.update_parameters()

    @override
    def opaque_parameter_ids(self) -> set[int]:
        """Return the parameters of the optimizers of the legacy
        strategy.

        Returns:
            set[int]: The ``id()`` of each parameter in the
            ``param_groups`` of each legacy optimizer.

        """
        optimizers, _ = self._mount()
        return {
            id(parameter)
            for optimizer in optimizers
            for group in optimizer.param_groups
            for parameter in group["params"]
        }

    @override
    def opaque_inners(self) -> list[tuple[Optimizer, Any]]:
        """Return the optimizers of the legacy strategy with their
        schedulers.

        Returns:
            ``list[tuple[Optimizer, Any]]``: One pair for each optimizer,
            in order. The scheduler at the same position completes the
            pair. When the legacy strategy returns fewer schedulers than
            optimizers, the last optimizers get ``None``.

        Raises:
            ValueError: If the legacy strategy returns more schedulers
                than optimizers.

        """
        optimizers, schedulers = self._mount()
        return list(zip(optimizers, schedulers, strict=True))
