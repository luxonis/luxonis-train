"""This module implements a metaclass for automatic registration of classes."""

import lightning.pytorch as pl
import torch
from luxonis_ml.utils.registry import Registry

import luxonis_train

CALLBACKS: Registry[type[pl.Callback]] = Registry(name="callbacks")
"""Registry for all callbacks."""

LOADERS: Registry[type["luxonis_train.utils.loaders.BaseLoaderTorch"]] = Registry(
    name="loaders"
)
"""Registry for all loaders."""

LOSSES: Registry[type["luxonis_train.attached_modules.BaseLoss"]] = Registry(
    name="losses"
)
"""Registry for all losses."""

METRICS: Registry[type["luxonis_train.attached_modules.BaseMetric"]] = Registry(
    name="metrics"
)
"""Registry for all metrics."""

MODELS: Registry[type["luxonis_train.models.BasePredefinedModel"]] = Registry(
    name="models"
)
"""Registry for all models."""

NODES: Registry[type["luxonis_train.nodes.BaseNode"]] = Registry(name="nodes")
"""Registry for all nodes."""

OPTIMIZERS: Registry[type[torch.optim.Optimizer]] = Registry(name="optimizers")
"""Registry for all optimizers."""

SCHEDULERS: Registry[type[torch.optim.lr_scheduler._LRScheduler]] = Registry(
    name="schedulers"
)
"""Registry for all schedulers."""

VISUALIZERS: Registry[type["luxonis_train.visualizers.BaseVisualizer"]] = Registry(
    "visualizers"
)
"""Registry for all visualizers."""
