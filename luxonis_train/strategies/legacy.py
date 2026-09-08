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
    """Mounts a strategy implementing the deprecated
    ``configure_optimizers`` API as opaque inner optimizers.

    The parameters the legacy strategy claims are excluded from the
    rule partition; everything it leaves unclaimed (frozen parameters
    included) still flows to the default tail, so such parameters keep
    training after unfreezing.
    """

    def __init__(self, legacy: Any):
        self._legacy = legacy
        self._mounted: tuple[list[Optimizer], list[Any]] | None = None

    @property
    def legacy_name(self) -> str:
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
        return []

    @override
    def get_base_configs(self) -> tuple[OptimizerConfig, SchedulerConfig]:
        get_base_configs = getattr(self._legacy, "get_base_configs", None)
        if get_base_configs is None:
            raise NotImplementedError
        return get_base_configs()

    @override
    def update_parameters(self) -> None:
        self._legacy.update_parameters()

    @override
    def opaque_parameter_ids(self) -> set[int]:
        optimizers, _ = self._mount()
        return {
            id(parameter)
            for optimizer in optimizers
            for group in optimizer.param_groups
            for parameter in group["params"]
        }

    @override
    def opaque_inners(self) -> list[tuple[Optimizer, Any]]:
        optimizers, schedulers = self._mount()
        return list(zip(optimizers, schedulers, strict=True))
