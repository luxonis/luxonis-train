"""This module implements a metaclass for automatic registration of classes."""

from typing import Any

import lightning.pytorch as pl
from luxonis_ml.utils.registry import Registry
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer

import luxonis_train as lt

CALLBACKS: Registry[type[pl.Callback]] = Registry(name="callbacks")
"""Registry for all callbacks."""

LOADERS: Registry[type["lt.utils.loaders.BaseLoaderTorch"]] = Registry(name="loaders")
"""Registry for all loaders."""

LOSSES: Registry[type["lt.attached_modules.BaseLoss[Any, Any]"]] = Registry(
    name="losses"
)
"""Registry for all losses."""

METRICS: Registry[type["lt.attached_modules.BaseMetric[Any, Any]"]] = Registry(
    name="metrics"
)
"""Registry for all metrics."""

MODELS: Registry[type["lt.models.BasePredefinedModel"]] = Registry(name="models")
"""Registry for all models."""

NODES: Registry[type["lt.nodes.BaseNode[Any, Any]"]] = Registry(name="nodes")
"""Registry for all nodes."""

OPTIMIZERS: Registry[type[Optimizer]] = Registry(name="optimizers")
"""Registry for all optimizers."""

SCHEDULERS: Registry[type[_LRScheduler]] = Registry(name="schedulers")
"""Registry for all schedulers."""

VISUALIZERS: Registry[type["lt.visualizers.BaseVisualizer[Any, Any]"]] = Registry(
    "visualizers"
)
"""Registry for all visualizers."""
