from abc import abstractmethod
from functools import cached_property
from inspect import Parameter

from torch import Tensor
from torchmetrics import Metric

from luxonis_train.attached_modules import BaseAttachedModule
from luxonis_train.utils import Labels, Packet
from luxonis_train.utils.registry import METRICS


class BaseMetric(BaseAttachedModule, Metric, register=False, registry=METRICS):
    """A base class for all metrics.

    This class defines the basic interface for all metrics. It utilizes
    automatic registration of defined subclasses to a L{METRICS}
    registry.
    """

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
