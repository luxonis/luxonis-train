from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from inspect import Parameter
from typing import Literal, get_args, get_origin, get_type_hints

from torch import Tensor
from torchmetrics import Metric
from typing_extensions import Annotated

from luxonis_train.attached_modules import BaseAttachedModule
from luxonis_train.typing import Labels, Packet
from luxonis_train.utils.registry import METRICS


@dataclass(kw_only=True, slots=True)
class State:
    """Marks an attribute that should be registered as a metric state.

    Upon initialization of a metric, all attributes of the metric that
    are marked with this class will be registered as metric states using
    the `add_state` method. The state will be accessible as an attribute
    of the metric instance.

    Metric state variables are either C{Tensor} or an empty list, which
    can be appended to by the metric. Each state variable must have a
    unique name associated with it. State variables are accessible as
    attributes of the metric i.e, if C{name} is C{"my_state"} then its
    value can be accessed from an instance C{metric} as
    C{metric.my_state}. Metric states behave like buffers and parameters
    of C{torch.nn.Module} as they are also updated when C{.to()} is
    called. Unlike parameters and buffers, metric states are not by
    default saved in the modules C{torch.nn.Module.state_dict}.

    @type name: str
    @param name: The name of the state variable. The variable will then
        be accessible at C{self.name}.
    @type default: Tensor | list
    @param default: Default value of the state; can either be a
        C{Tensor} or an empty list. The state will be reset to this
        value when C{self.reset()} is called.
    @type dist_reduce_fx: Literal["sum", "mean", "cat", "min", "max"] |
        Callable[[Tensor], Tensor] | Callable[[list[Tensor]], Tensor] |
        None
    @param dist_reduce_fx: Function to reduce state across multiple
        processes in distributed mode. If value is C{"sum"}, C{"mean"},
        C{"cat"}, C{"min"} or C{"max"} we will use C{torch.sum},
        C{torch.mean}, C{torch.cat}, C{torch.min} and C{torch.max}
        respectively, each with argument C{dim=0}. Note that the
        C{"cat"} reduction only makes sense if the state is a list, and
        not a tensor. The user can also pass a custom function in this
        parameter.
    @type persistent: bool
    @param persistent: whether the state will be saved as part of the
        modules C{state_dict}. Default is C{False}.
    """

    default: Tensor | list
    dist_reduce_fx: (
        Literal["sum", "mean", "cat", "min", "max"]
        | Callable[[Tensor], Tensor]
        | Callable[[list[Tensor]], Tensor]
        | None
    ) = None
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
                state_config = next(
                    (arg for arg in type_args if isinstance(arg, State)), None
                )
                if state_config is not None:
                    self.add_state(
                        attr_name,
                        default=state_config.default,
                        dist_reduce_fx=state_config.dist_reduce_fx,
                        persistent=state_config.persistent,
                    )

    @abstractmethod
    def update(self, *args: Tensor | list[Tensor]) -> None:
        """Updates the inner state of the metric.

        @type args: Unpack[Ts]
        @param args: Prepared inputs from the L{prepare} method.
        """
        ...

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
        ...

    @cached_property
    def _signature(self) -> dict[str, Parameter]:
        return self._get_signature(self.update)

    def run_update(self, inputs: Packet[Tensor], labels: Labels) -> None:
        """Calls the metric's update method.

        Validates and prepares the inputs, then calls the metric's
        update method.

        @type inputs: Packet[Tensor]
        @param inputs: The outputs of the model.
        @type labels: Labels
        @param labels: The labels of the model. @raises
            L{IncompatibleException}: If the inputs are not compatible
            with the module.
        """
        self.update(**self.get_parameters(inputs, labels))
