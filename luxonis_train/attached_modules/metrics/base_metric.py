from abc import abstractmethod

from torch import Tensor
from torchmetrics import Metric
from typing_extensions import TypeVarTuple, Unpack

from luxonis_train.attached_modules import BaseAttachedModule
from luxonis_train.utils.registry import METRICS
from luxonis_train.utils.types import Labels, Packet

Ts = TypeVarTuple("Ts")


class BaseMetric(
    BaseAttachedModule[Unpack[Ts]],
    Metric,
    register=False,
    registry=METRICS,
):
    """A base class for all metrics.

    This class defines the basic interface for all metrics. It utilizes automatic
    registration of defined subclasses to a L{METRICS} registry.
    """

    @abstractmethod
    def update(self, *args: Unpack[Ts]) -> None:
        """Updates the inner state of the metric.

        @type args: Unpack[Ts]
        @param args: Prepared inputs from the L{prepare} method.
        """
        ...

    @abstractmethod
    def compute(self) -> Tensor | tuple[Tensor, dict[str, Tensor]] | dict[str, Tensor]:
        """Computes the metric.

        @rtype: Tensor | tuple[Tensor, dict[str, Tensor]] | dict[str, Tensor]
        @return: The computed metric. Can be one of:
           - A single Tensor.
           - A tuple of a Tensor and a dictionary of submetrics.
           - A dictionary of submetrics. If this is the case, then the metric
              cannot be used as the main metric of the model.
        """
        ...

    def run_update(self, outputs: Packet[Tensor], labels: Labels) -> None:
        """Calls the metric's update method.

        Validates and prepares the inputs, then calls the metric's update method.

        @type outputs: Packet[Tensor]
        @param outputs: The outputs of the model.
        @type labels: Labels
        @param labels: The labels of the model. @raises L{IncompatibleException}: If the
            inputs are not compatible with the module.
        """
        self.validate(outputs, labels)
        self.update(*self.prepare(outputs, labels))
