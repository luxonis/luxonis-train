from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from inspect import Parameter
from types import EllipsisType
from typing import (
    Annotated,
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
from luxonis_train.typing import Labels, Packet, get_signature


@dataclass(kw_only=True, slots=True)
class MetricState:
    """Marks an attribute that should be registered as a metric state.
    Intended to be used as a type hint for class attributes using the
    `Annotated` type.

    Upon initialization of a metric, all attributes of the metric that
    are marked as metric states will be registered using the
    `add_state` method. The state will be accessible as an attribute
    of the metric instance.

    Metric state variables are either C{Tensor} or an empty list, which
    can be appended to by the metric.  Metric states behave like buffers
    and parameters of C{nn.Module} as they are also updated when C{.to()}
    is called. Unlike parameters and buffers, metric states are not by
    default saved in the modules C{nn.Module.state_dict}.

    The metric state variables are automatically reset to their default
    values when the metric's C{reset()} method is called.

    Example usage::

        class MyMetric(BaseMetric):
            true_positives: Annotated[Tensor, MetricState(default=0)]
            false_positives: Annotated[Tensor, MetricState(default=0)]
            total: Annotated[Tensor, MetricState(default=0)]

    @keyword name: The name of the state variable. The variable will then
        be accessible at C{self.name}.
    @keyword default: Default value of the state; can either be a
        C{Tensor} or an empty list. The state will be reset to this
        value when C{self.reset()} is called. If the default value is a
        float, it will be converted to a C{Tensor}.
    @keyword dist_reduce_fx: Function to reduce state across multiple
        processes in distributed mode. If value is C{"sum"}, C{"mean"},
        C{"cat"}, C{"min"} or C{"max"} we will use C{torch.sum},
        C{torch.mean}, C{torch.cat}, C{torch.min} and C{torch.max}
        respectively, each with argument C{dim=0}. Note that the
        C{"cat"} reduction only makes sense if the state is a list, and
        not a tensor. The user can also pass a custom function in this
        parameter.
        If not specified, the default is C{"sum"} if the default value
        is a tensor, and C{"cat"} if the default value is a list.
    @keyword persistent: Whether the state will be saved as part of the
        modules C{state_dict}. Default is C{False}.
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
    automatic registration of defined subclasses to a L{METRICS}
    registry.
    """

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
        """Updates the inner state of the metric.

        @type args: Unpack[Ts]
        @param args: Prepared inputs from the L{prepare} method.
        """
        super().update(*args)

    @abstractmethod
    def compute(
        self,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]] | dict[str, Tensor]:
        """Computes the metric.

        @rtype: Tensor | tuple[Tensor, dict[str, Tensor]] | dict[str, Tensor]
        @return: The computed metric. Can be one of:
           - A single Tensor.
           - A tuple of a Tensor and a dictionary of sub-metrics.
           - A dictionary of sub-metrics. If this is the case, then the metric
              cannot be used as the main metric of the model.
        """
        return super().compute()

    @cached_property
    def _signature(self) -> dict[str, Parameter]:
        return get_signature(self.update)

    def run_update(self, inputs: Packet[Tensor], labels: Labels) -> None:
        """Calls the metric's update method.

        Validates and prepares the inputs, then calls the metric's
        update method.

        @type inputs: Packet[Tensor]
        @param inputs: The outputs of the model.
        @type labels: Labels
        @param labels: The labels of the model. @raises
            L{IncompatibleError}: If the inputs are not compatible with
            the module.
        """
        self.update(**self.get_parameters(inputs, labels))
