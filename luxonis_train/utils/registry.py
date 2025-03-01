"""This module implements a metaclass for automatic registration of
classes."""

from typing import TypeVar

import lightning.pytorch as pl
from luxonis_ml.utils.registry import Registry
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer

import luxonis_train as lxt

MODELS: Registry[type["lxt.config.BasePredefinedModel"]] = Registry(
    name="models"
)
"""Registry for all models."""

CALLBACKS: Registry[type[pl.Callback]] = Registry(name="callbacks")
"""Registry for all callbacks."""

LOADERS: Registry[type["lxt.loaders.BaseLoaderTorch"]] = Registry(
    name="loaders"
)
"""Registry for all loaders."""

LOSSES: Registry[type["lxt.attached_modules.BaseLoss"]] = Registry(
    name="losses"
)
"""Registry for all losses."""

METRICS: Registry[type["lxt.attached_modules.BaseMetric"]] = Registry(
    name="metrics"
)
"""Registry for all metrics."""

NODES: Registry[type["lxt.nodes.BaseNode"]] = Registry(name="nodes")
"""Registry for all nodes."""

OPTIMIZERS: Registry[type[Optimizer]] = Registry(name="optimizers")
"""Registry for all optimizers."""

SCHEDULERS: Registry[type[LRScheduler]] = Registry(name="schedulers")
"""Registry for all schedulers."""

STRATEGIES: Registry[type["lxt.strategies.BaseTrainingStrategy"]] = Registry(
    name="strategies"
)
"""Registry for all strategies."""

VISUALIZERS: Registry[type["lxt.visualizers.BaseVisualizer"]] = Registry(
    "visualizers"
)
"""Registry for all visualizers."""


T = TypeVar("T")


def from_registry(registry: Registry[type[T]], key: str, *args, **kwargs) -> T:
    """Get an instance of the class registered under the given key.

    @type registry: Registry[type[T]]
    @param registry: Registry to get the class from.
    @type key: str
    @param key: Key to get the class for.
    @rtype: T
    @return: Instance of the class registered under the given key.
    """
    return registry.get(key)(*args, **kwargs)
