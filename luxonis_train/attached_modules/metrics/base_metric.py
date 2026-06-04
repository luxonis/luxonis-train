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


@dataclass(kw_only=True, slots=True)
class MetricState:
    """Marks an attribute that should be registered as a metric state.
    Intended to be used as a type hint for class attributes using the
    `Annotated` type.

    Upon initialization of a metric, all attributes of the metric that
    are marked as metric states will be registered using the
    `add_state` method. The state will be accessible as an attribute
    of the metric instance.

    Metric state variables are either ``Tensor`` or an empty list, which
    can be appended to by the metric.  Metric states behave like buffers
    and parameters of ``nn.Module`` as they are also updated when ``.to()``
    is called. Unlike parameters and buffers, metric states are not by
    default saved in the modules ``nn.Module.state_dict``.

    The metric state variables are automatically reset to their default
    values when the metric's ``reset()`` method is called.

    Example usage::

        class MyMetric(BaseMetric):
            true_positives: Annotated[Tensor, MetricState(default=0)]
            false_positives: Annotated[Tensor, MetricState(default=0)]
            total: Annotated[Tensor, MetricState(default=0)]

    Keyword Args:
        name (Any): The name of the state variable. The variable will then be accessible at
            ``self.name``.
        default (Any): Default value of the state; can either be a ``Tensor`` or an empty list.
            The state will be reset to this value when ``self.reset()`` is called. If the
            default value is a float, it will be converted to a ``Tensor``.
        dist_reduce_fx (Any): Function to reduce state across multiple processes in distributed
            mode. If value is ``"sum"``, ``"mean"``, ``"cat"``, ``"min"`` or ``"max"`` we will
            use ``torch.sum``, ``torch.mean``, ``torch.cat``, ``torch.min`` and ``torch.max``
            respectively, each with argument ``dim=0``. Note that the ``"cat"`` reduction only
            makes sense if the state is a list, and not a tensor. The user can also pass a
            custom function in this parameter. If not specified, the default is ``"sum"`` if
            the default value is a tensor, and ``"cat"`` if the default value is a list.
        persistent (Any): Whether the state will be saved as part of the modules
            ``state_dict``. Default is ``False``.

    """

    default: Tensor | Number | list | None = None
    dist_reduce_fx: (
        Literal["sum", "mean", "cat", "min", "max"]
        | Callable[[Tensor], Tensor]
        | Callable[[list[Tensor]], Tensor]
        | EllipsisType
        | None
    ) = ...
    persistent: bool = False


class BaseMetric(BaseAttachedModule, Metric, register=False, registry=METRICS):
    """A base class for all metrics.

    This class defines the basic interface for all metrics. It utilizes
    automatic registration of defined subclasses to a `METRICS`
    registry.

    """

    predefined_model_params_aliases: ClassVar[dict[str, str]] = {}

    @classmethod
    def get_predefined_model_params_aliases(
        cls, task: Task | None = None
    ) -> dict[str, str]:
        return cls.predefined_model_params_aliases

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        hints = get_type_hints(self.__class__, include_extras=True)

        for attr_name, attr_type in hints.items():
            if get_origin(attr_type) is Annotated:
                type_args = get_args(attr_type)
                main_type = type_args[0]

                state = next(
                    (arg for arg in type_args if isinstance(arg, MetricState)),
                    None,
                )
                if state is not None:
                    default = state.default
                    if default is None:
                        if main_type is Tensor:
                            default = 0.0
                        elif getattr(main_type, "__origin__", None) is list:
                            default = []
                        else:
                            raise ValueError(
                                f"Unsupported type of a metric state: `{main_type}`"
                            )
                    if isinstance(default, Number):
                        default = torch.tensor(default)

                    dist_reduce_fx = state.dist_reduce_fx
                    if dist_reduce_fx is ...:
                        if isinstance(default, list):
                            dist_reduce_fx = "cat"
                        else:
                            dist_reduce_fx = "sum"

                    self.add_state(
                        attr_name,
                        default=default,
                        dist_reduce_fx=dist_reduce_fx,
                        persistent=state.persistent,
                    )

    @abstractmethod
    def update(self, *args: Tensor | list[Tensor]) -> None:
        """Update the inner state of the metric.

        Args:
            *args (Unpack[Ts]): Prepared inputs from the `prepare` method.

        """
        super().update(*args)

    @abstractmethod
    def compute(
        self,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]] | dict[str, Tensor]:
        """Compute the metric.

        Returns:
            Tensor | tuple[Tensor, dict[str, Tensor]] | dict[str, Tensor]: The computed metric. Can
                be one of: - A single Tensor. - A tuple of a Tensor and a dictionary of
                sub-metrics. - A dictionary of sub-metrics. If this is the case, then the metric
                cannot be used as the main metric of the model.

        """
        return super().compute()

    @cached_property
    def _signature(self) -> dict[str, Parameter]:
        return get_signature(self.update)

    def run_update(self, inputs: Packet[Tensor], labels: Labels) -> None:
        """Call the metric's update method.

        Validates and prepares the inputs, then calls the metric's
        update method.

        Args:
            inputs (Packet[Tensor]): The outputs of the model.
            labels (Labels): The labels of the model.

        Raises:
            IncompatibleError: If the inputs are not compatible with the module.

        """
        self.update(**self.get_parameters(inputs, labels))
