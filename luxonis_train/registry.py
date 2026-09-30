"""Registries that map component names from a config to their classes.

Most components register through their base-class metaclass. The key is
the class name unless ``register_name`` overrides it. Predefined models
also use versioned keys such as ``DetectionModel:v1`` and alias the
latest version under the bare name and ``:latest``.

"""

from typing import TYPE_CHECKING, Any, TypeVar

from luxonis_ml.utils import Registry

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

_INTERNAL: dict[str, Any] = {}


T = TypeVar("T")


def from_registry(registry: Registry[type[T]], key: str, *args, **kwargs) -> T:
    """Create an instance of the class registered under a name.

    The function gets the class for ``key`` from ``registry`` and calls
    it with ``args`` and ``kwargs``. The registry raises ``KeyError``
    when it has no class under ``key``.

    Args:
        registry (``Registry[type[T]]``): The registry to search, for
            example `NODES` or `OPTIMIZERS`.
        key (str): The name of the registered class.
        *args (``Any``): The positional arguments for the constructor.
        **kwargs (``Any``): The keyword arguments for the constructor.

    Returns:
        ``T``: The result of the call. It is a new instance of the
        registered class, unless the class is a factory such as
        `ConfusionMatrix`.

    Example:
        >>> import torch
        >>> from luxonis_train.registry import OPTIMIZERS, from_registry
        >>> params = [torch.nn.Parameter(torch.zeros(2))]
        >>> optimizer = from_registry(OPTIMIZERS, "SGD", params, lr=0.1)
        >>> type(optimizer).__name__, optimizer.defaults["lr"]
        ('SGD', 0.1)

    """
    return registry.get(key)(*args, **kwargs)
