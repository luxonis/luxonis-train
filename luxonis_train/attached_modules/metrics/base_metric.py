"""The base class of all metrics and the marker for metric states.

`BaseMetric` combines `BaseAttachedModule` with the ``torchmetrics``
``Metric`` class. `MetricState` marks the class attributes that
`BaseMetric` registers as metric states. ``DistReduceFx`` is the type of
the reduction that merges one state across processes, and
``MetricResult`` is the type of the result of `BaseMetric.compute`.

"""

from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from inspect import Parameter
from types import EllipsisType
from typing import (
    Annotated,
    ClassVar,
    Literal,
    get_args,
    get_origin,
    get_type_hints,
)

import torch
from torch import Tensor
from torch.types import Number
from torchmetrics import Metric

from luxonis_train.attached_modules import BaseAttachedModule
from luxonis_train.registry import METRICS
from luxonis_train.tasks import Task
from luxonis_train.typing import Labels, Packet
from luxonis_train.utils import get_signature

MetricResult = Tensor | tuple[Tensor, dict[str, Tensor]] | dict[str, Tensor]

DistReduceFx = (
    Literal["sum", "mean", "cat", "min", "max"]
    | Callable[[Tensor], Tensor]
    | Callable[[list[Tensor]], Tensor]
    | None
)


@dataclass(kw_only=True, slots=True)
class MetricState:
    """Marker for a class attribute that becomes a metric state.

    Put an instance into the ``Annotated`` type hint of a class
    attribute of a `BaseMetric` subclass. `BaseMetric.__init__` finds
    each such attribute in the class and in its base classes. It
    registers the attribute with the ``add_state`` method of
    ``torchmetrics``, so the state is an attribute of the metric
    instance. The first argument of ``Annotated`` is the type of the
    state.

    ``torchmetrics`` treats a state like a buffer: ``.to()`` moves it.
    ``reset`` gives a tensor state a copy of its default and empties a
    list state.

    Attributes:
        default (``Tensor | Number | list | None``): The value of the
            state after initialization and after ``reset``. A number
            becomes a zero-dimensional tensor, so ``0`` gives an
            ``int64`` state and ``0.0`` a ``float32`` state. ``None``
            selects ``0.0`` for a ``Tensor`` attribute and ``[]`` for a
            ``list[...]`` attribute. For an attribute of any other type,
            ``None`` makes `BaseMetric.__init__` raise ``ValueError``.
            A list default must be empty, else ``add_state`` raises
            ``ValueError``. **Each instance receives a tensor or list
            default as it is, not a copy.** All instances of the class
            then share one state object. An in-place change such as
            ``+=`` or ``append`` in one metric also changes the others.
            ``reset`` gives one metric a new tensor, so that metric no
            longer shares a tensor state. ``reset`` empties a list state
            in place, so the list stays shared and empties for all
            metrics. A number or ``None`` gives each instance its own
            state.
        dist_reduce_fx (``DistReduceFx | EllipsisType``): The reduction
            that merges the state of all processes. A string selects
            ``torch.sum``, ``torch.mean``, ``torch.cat``, ``torch.min``,
            or ``torch.max`` over dimension ``0``. A callable receives
            the gathered state: the tensor states stacked along a new
            first dimension, or one list with the items of all
            processes. ``None`` keeps the gathered state as it is. The
            default ``...`` selects ``"cat"`` for a list state and
            ``"sum"`` for a tensor state.
        persistent (bool): Whether the ``state_dict`` of the metric
            holds the state.

    Example:
        A subclass with two tensor states. ``register=False`` keeps the
        class out of the `METRICS` registry.

        >>> from typing import Annotated
        >>> import torch
        >>> from torch import Tensor
        >>> class PositiveRate(BaseMetric, register=False):
        ...     positives: Annotated[Tensor, MetricState()]
        ...     total: Annotated[Tensor, MetricState(default=0)]
        ...
        ...     def update(self, predictions: Tensor) -> None:
        ...         self.positives += (predictions > 0).sum()
        ...         self.total += predictions.numel()
        ...
        ...     def compute(self) -> Tensor:
        ...         return self.positives / self.total
        >>> metric = PositiveRate()
        >>> metric.update(torch.tensor([1.0, -1.0, 2.0, 3.0]))
        >>> metric.compute().item()
        0.75
        >>> metric.reset()
        >>> metric.positives.item(), metric.total.item()
        (0.0, 0)

    """

    default: Tensor | Number | list | None = None
    dist_reduce_fx: DistReduceFx | EllipsisType = ...
    persistent: bool = False


class BaseMetric(BaseAttachedModule, Metric, register=False, registry=METRICS):
    """Base class for all metrics.

    A metric is a `BaseAttachedModule` and a ``torchmetrics`` ``Metric``.
    Every subclass registers itself in the `METRICS` registry under its
    class name, unless its class statement passes ``register=False``. A
    config names a registered metric by that string. A subclass
    implements `update` and `compute`. It declares its states with
    `MetricState`, or with ``add_state`` in its ``__init__``. The
    example of `MetricState` shows a complete subclass.

    `run_update` fills the parameters of `update` by their names:

    - ``predictions``, or another name that starts with ``pred`` and
      has no underscore, selects the main output of the task.
    - Another name that starts with ``pred`` selects the packet key
      after the first underscore, so ``pred_boundingbox`` selects
      ``boundingbox``.
    - ``target``, or another name that starts with ``target`` and has
      no underscore, selects the single label that the task requires.
      ``target_<label>`` selects the label ``<label>``. Both look the
      label up as ``<task_name>/<label>``, with the ``task_name`` of
      the node.
    - Any other name selects the packet key of that name.

    The trainer calls `run_update` on each validation and test batch.
    At the end of the epoch, it calls `compute` and logs the images of
    `get_artifacts`. It then calls ``reset`` and logs the values that
    `get_loggable_values` selects.

    Two metrics are equal only when they are the same object, and the
    hash of a metric is its ``id``. In ``torchmetrics``, ``==`` builds a
    new composed metric, and the hash reads the states.

    """

    predefined_model_params_aliases: ClassVar[dict[str, str]] = {}

    @classmethod
    def get_predefined_model_params_aliases(
        cls, task: Task | None = None
    ) -> dict[str, str]:
        """Return the constructor names of predefined model parameters.

        A predefined model can add ``per_class_metrics`` to the
        ``params`` of each of its metrics. When `Nodes` builds the
        metric, it looks the key up in the returned dictionary and
        passes the value under the parameter name that it finds. When
        the dictionary has no such key, `Nodes` drops the value and logs
        a warning. It drops a ``None`` value without a lookup.

        This implementation ignores ``task`` and returns the class
        attribute ``predefined_model_params_aliases``. The attribute is
        empty on `BaseMetric`, and `MIoU` maps ``per_class_metrics`` to
        ``per_class``. `MeanAveragePrecision` is not a subclass. It
        defines its own method, because its mapping depends on the task.

        Args:
            task (Task | None): The task of the node that the metric
                attaches to, or ``None`` when the node has no task.

        Returns:
            dict[str, str]: The predefined model parameter names, mapped
            to the parameter names of the constructor.

        Example:
            >>> from luxonis_train.attached_modules.metrics import MIoU
            >>> BaseMetric.get_predefined_model_params_aliases()
            {}
            >>> MIoU.get_predefined_model_params_aliases()
            {'per_class_metrics': 'per_class'}

        """
        return cls.predefined_model_params_aliases

    def __init__(self, **kwargs):
        """Initialize the metric and register its metric states.

        The method reads the type hints of the class and of its base
        classes. It registers each attribute whose ``Annotated`` hint
        holds a `MetricState` with the ``add_state`` method of
        ``torchmetrics``. `MetricState` describes how the default value
        and the reduction of a state follow from the marker.

        Args:
            **kwargs (``Any``): Keyword arguments forwarded to
                `BaseAttachedModule`, such as ``node``. The remaining
                arguments reach the ``torchmetrics`` ``Metric``, which
                accepts only its own options, such as
                ``sync_on_compute``, and raises ``ValueError`` for any
                other name.

        Raises:
            ValueError: When a metric state has no ``default`` and its
                type is neither ``Tensor`` nor ``list[...]``. Also when
                the ``default`` of a metric state is a list that is not
                empty.

        """
        super().__init__(**kwargs)

        hints = get_type_hints(self.__class__, include_extras=True)

        for attr_name, attr_type in hints.items():
            self._register_metric_state(attr_name, attr_type)

    def _register_metric_state(
        self, attr_name: str, attr_type: object
    ) -> None:
        if get_origin(attr_type) is not Annotated:
            return
        type_args = get_args(attr_type)
        state = next(
            (arg for arg in type_args if isinstance(arg, MetricState)), None
        )
        if state is None:
            return
        default = self._metric_state_default(type_args[0], state.default)
        self.add_state(
            attr_name,
            default=default,
            dist_reduce_fx=self._metric_state_reducer(
                default, state.dist_reduce_fx
            ),
            persistent=state.persistent,
        )

    @staticmethod
    def _metric_state_default(
        main_type: object, default: Tensor | Number | list | None
    ) -> Tensor | list:
        if default is None:
            if main_type is Tensor:
                default = 0.0
            elif getattr(main_type, "__origin__", None) is list:
                default = []
            else:
                raise ValueError(
                    f"Unsupported type of a metric state: `{main_type}`"
                )
        return (
            torch.tensor(default) if isinstance(default, Number) else default
        )

    @staticmethod
    def _metric_state_reducer(
        default: Tensor | list,
        reducer: DistReduceFx | EllipsisType,
    ) -> DistReduceFx:
        if reducer is not ...:
            return reducer
        return "cat" if isinstance(default, list) else "sum"

    @abstractmethod
    def update(self, *args: Tensor | list[Tensor]) -> None:
        """Add the data of one batch to the metric states.

        An implementation declares one named parameter for each input.
        `run_update` fills the parameters by name, as the class
        docstring describes, and passes them as keyword arguments. The
        implementation adds the batch to the states, and `compute`
        derives the value from them.

        ``torchmetrics`` wraps the method. Each call clears the cached
        result of `compute` and runs with gradients disabled.

        Args:
            *args (``Tensor | list[Tensor]``): The inputs of the batch.
                An implementation replaces them with named parameters.

        """
        super().update(*args)

    @abstractmethod
    def compute(
        self,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]] | dict[str, Tensor]:
        """Compute the value of the metric from its states.

        ``torchmetrics`` wraps the method. The wrapper warns when no
        `update` ran since the last ``reset``. In a distributed run, it
        syncs the states across processes. It squeezes each one-element
        tensor to a scalar and clones the result. It caches the result
        until the next `update` or ``reset``.

        Returns:
            ``Tensor | tuple[Tensor, dict[str, Tensor]] | dict[str, Tensor]``:
            The result in one of three forms:

            - The main value as a ``Tensor``.
            - A tuple of the main value and a dictionary of sub-metrics.
            - A dictionary of sub-metrics only. The trainer then logs no
              value under the name of the metric, so the metric cannot
              be the main metric.

        """
        return super().compute()

    def get_loggable_values(
        self,
        values: MetricResult,
    ) -> MetricResult:
        """Select the part of a computed result that the trainer logs.

        The trainer passes the result of `compute` through this method
        before it logs and prints the values. `get_artifacts` receives
        the full result. This implementation returns ``values``
        unchanged. `PrecisionRecallCurve` overrides it, because
        `PrecisionRecallCurve.compute` returns whole curves.

        Args:
            values (``Tensor | tuple[Tensor, dict[str, Tensor]] | dict[str, Tensor]``):
                The result of `compute`.

        Returns:
            ``Tensor | tuple[Tensor, dict[str, Tensor]] | dict[str, Tensor]``:
            The values to log, in one of the forms of `compute`.

        """
        return values

    def get_artifacts(
        self,
        values: MetricResult,
    ) -> dict[str, Tensor]:
        """Render images from a computed result.

        At the end of an evaluation epoch, the trainer calls this method
        on the main process with the full result of `compute`. It skips
        the call during the sanity check. It logs each returned image to
        the tracker. It logs a warning for an item that is not a tensor
        with three dimensions, and skips that item. When the method
        raises or returns no dictionary, the trainer logs the problem
        and continues. This implementation returns an empty dictionary.

        Args:
            values (``Tensor | tuple[Tensor, dict[str, Tensor]] | dict[str, Tensor]``):
                The result of `compute`.

        Returns:
            ``dict[str, Tensor]``: The images of shape ``[C, H, W]``,
            keyed by the names that `get_artifact_names` returns.

        """
        return {}

    def get_artifact_names(self) -> tuple[str, ...]:
        """Return the names of the images that `get_artifacts` renders.

        `LuxonisLightningModule.get_mlflow_logging_keys` uses the names
        to list the artifact paths of a run without rendering the
        images. This implementation returns an empty tuple.

        Returns:
            ``tuple[str, ...]``: The keys of the dictionary that
            `get_artifacts` returns.

        """
        return ()

    def __eq__(self, other: object) -> bool:
        """Return whether ``other`` is this metric object.

        Args:
            other (object): The object to compare with.

        Returns:
            bool: ``True`` only when ``other`` is ``self``.

        """
        return self is other

    def __hash__(self) -> int:
        """Return the ``id`` of the metric as its hash.

        The hash stays the same when the states change.

        Returns:
            int: ``id(self)``.

        """
        return id(self)

    @cached_property
    def _signature(self) -> dict[str, Parameter]:
        """The parameters of `update` that `run_update` fills.

        `get_signature` leaves out ``self`` and ``kwargs``.

        """
        return get_signature(self.update)

    def run_update(self, inputs: Packet[Tensor], labels: Labels) -> None:
        """Select the inputs of `update` from a batch and call it.

        `BaseAttachedModule.get_parameters` picks a value for each
        parameter of `update` by its name, as the class docstring
        describes. It clones every tensor that it picks, so `update`
        cannot change ``inputs`` or ``labels``.

        When a value is missing, a parameter annotated with ``| None``
        receives ``None``, even when it has a default value. Another
        parameter with a default value keeps the default. For any other
        parameter, the lookup raises ``RuntimeError``. It also raises
        ``RuntimeError`` for a ``target`` name without an underscore
        when the task requires more than one label. A value that does
        not match the annotation of its parameter raises ``TypeError``.

        Args:
            inputs (``Packet[Tensor]``): The output packet of the node.
            labels (``Labels``): The labels of the batch, keyed
                ``<task_name>/<label>``.

        """
        self.update(**self.get_parameters(inputs, labels))
