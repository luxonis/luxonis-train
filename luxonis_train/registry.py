"""This module implements a metaclass for automatic registration of
classes."""

from typing import TYPE_CHECKING, TypeVar

from luxonis_ml.utils.registry import Registry

if TYPE_CHECKING:
    import lightning.pytorch as pl
    from torch.optim.lr_scheduler import LRScheduler
    from torch.optim.optimizer import Optimizer

    import luxonis_train as lxt

CALLBACKS: Registry[type["pl.Callback"]] = Registry(name="callbacks")

LOADERS: Registry[type["lxt.BaseLoaderTorch"]] = Registry(name="loaders")

LOSSES: Registry[type["lxt.BaseLoss"]] = Registry(name="losses")

METRICS: Registry[type["lxt.BaseMetric"]] = Registry(name="metrics")

MODELS: Registry[type["lxt.BasePredefinedModel"]] = Registry(name="models")

NODES: Registry[type["lxt.BaseNode"]] = Registry(name="nodes")

OPTIMIZERS: Registry[type["Optimizer"]] = Registry(name="optimizers")

SCHEDULERS: Registry[type["LRScheduler"]] = Registry(name="schedulers")

STRATEGIES: Registry[type["lxt.BaseTrainingStrategy"]] = Registry(
    name="strategies"
)

VISUALIZERS: Registry[type["lxt.BaseVisualizer"]] = Registry("visualizers")


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
