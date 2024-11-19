"""This module implements a metaclass for automatic registration of
classes."""

import lightning.pytorch as pl
from luxonis_ml.utils.registry import Registry
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer

import luxonis_train as lt

CALLBACKS: Registry[type[pl.Callback]] = Registry(name="callbacks")
"""Registry for all callbacks."""

LOADERS: Registry[type["lt.loaders.BaseLoaderTorch"]] = Registry(
    name="loaders"
)
"""Registry for all loaders."""

LOSSES: Registry[type["lt.attached_modules.BaseLoss"]] = Registry(
    name="losses"
)
"""Registry for all losses."""

METRICS: Registry[type["lt.attached_modules.BaseMetric"]] = Registry(
    name="metrics"
)
"""Registry for all metrics."""

NODES: Registry[type["lt.nodes.BaseNode"]] = Registry(name="nodes")
"""Registry for all nodes."""

OPTIMIZERS: Registry[type[Optimizer]] = Registry(name="optimizers")
"""Registry for all optimizers."""

SCHEDULERS: Registry[type[LRScheduler]] = Registry(name="schedulers")
"""Registry for all schedulers."""

STRATEGIES: Registry[type["lt.strategies.BaseTrainingStrategy"]] = Registry(
    name="strategies"
)
"""Registry for all strategies."""

VISUALIZERS: Registry[type["lt.visualizers.BaseVisualizer"]] = Registry(
    "visualizers"
)
"""Registry for all visualizers."""
