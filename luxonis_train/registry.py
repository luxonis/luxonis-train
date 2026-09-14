"""The registries that map a name in a config to a class.

Each registry is a ``luxonis_ml`` ``Registry``. A base class that
names a registry registers each subclass when Python creates the
subclass. The key is the class name, or the ``register_name`` of the
class statement. The keys of `MODELS` also carry a version, see below.
A class statement with ``register=False`` skips the registration. Thus
a config refers to a component by name and does not import it.

Some registered classes have no such base class. The package registers
them explicitly:

- The callbacks, in `luxonis_train.callbacks`.
- The optimizers, in `luxonis_train.optimizers`.
- The schedulers, in `luxonis_train.schedulers`.
- The metric factories `ConfusionMatrix` and `MeanAveragePrecision`.

The registries are:

- `CALLBACKS`: the callbacks, for the ``name`` of each entry in
  ``trainer.callbacks``. The values are subclasses of ``pl.Callback``,
  from Lightning and from this package.
- `LOADERS`: the subclasses of `BaseLoaderTorch`, for ``loader.name``.
- `LOSSES`: the subclasses of `BaseLoss`, for the ``losses`` of a
  node.
- `METRICS`: the subclasses of `BaseMetric` and the two metric
  factories, for the ``metrics`` of a node.
- `MODELS`: the concrete subclasses of `BasePredefinedModel`, for
  ``model.predefined_model.name``. Each model has the key
  ``<name>:v<N>``. The highest version of a model also has the keys
  ``<name>`` and ``<name>:latest``.
- `NODES`: the subclasses of `BaseNode`, for the ``name`` of each entry
  in ``model.nodes``.
- `OPTIMIZERS`: a fixed list of optimizers from ``torch.optim``, for
  ``trainer.optimizer.name``. The list does not contain every optimizer
  of ``torch.optim``. The optimizer names of the ``finetuning`` entries
  of a node and of ``exporter.aimet`` also use this registry.
- `SCHEDULERS`: a fixed list of schedulers from
  ``torch.optim.lr_scheduler``, for ``trainer.scheduler.name``. The
  scheduler names of the ``finetuning`` entries of a node and of
  ``exporter.aimet`` also use this registry.
- `STRATEGIES`: the subclasses of `BaseTrainingStrategy`, for
  ``trainer.training_strategy.name``.
- `VISUALIZERS`: the subclasses of `BaseVisualizer`, for the
  ``visualizers`` of a node.

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
