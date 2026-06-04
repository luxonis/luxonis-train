"""Registry objects and helpers for Luxonis Train components.

Attributes:
    CALLBACKS (Registry[type[pl.Callback]]): Registered Lightning callback
        classes.
    LOADERS (Registry[type[lxt.BaseLoaderTorch]]): Registered loader classes.
    LOSSES (Registry[type[lxt.BaseLoss]]): Registered loss classes.
    METRICS (Registry[type[lxt.BaseMetric]]): Registered metric classes.
    MODELS (Registry[type[lxt.BasePredefinedModel]]): Registered model classes.
    NODES (Registry[type[lxt.BaseNode]]): Registered node classes.
    OPTIMIZERS (Registry[type[Optimizer]]): Registered optimizer classes.
    SCHEDULERS (Registry[type[LRScheduler]]): Registered scheduler classes.
    STRATEGIES (Registry[type[lxt.BaseTrainingStrategy]]): Registered strategy
        classes.
    VISUALIZERS (Registry[type[lxt.BaseVisualizer]]): Registered visualizer
        classes.

"""

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

    Args:
        registry (Registry[type[T]]): Registry to get the class from.
        key (str): Key to get the class for.
        *args (Any): Positional arguments forwarded to the registered class.
        **kwargs (Any): Keyword arguments forwarded to the registered class.

    Returns:
        T: Instance of the class registered under the given key.

    """
    return registry.get(key)(*args, **kwargs)
