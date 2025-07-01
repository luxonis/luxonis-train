from abc import abstractmethod
from functools import cached_property
from inspect import Parameter

from torch import Tensor

from luxonis_train.attached_modules import BaseAttachedModule
from luxonis_train.registry import LOSSES
from luxonis_train.typing import Labels, Packet


class BaseLoss(BaseAttachedModule, register=False, registry=LOSSES):
    """A base class for all loss functions.

    This class defines the basic interface for all loss functions. It
    utilizes automatic registration of defined subclasses to a L{LOSSES}
    registry.
    """

    @abstractmethod
    def forward(
        self, *args: Tensor | list[Tensor]
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        """Forward pass of the loss function.

        @type args: Unpack[Ts]
        @param args: Prepared inputs from the L{prepare} method.
        @rtype: Tensor | tuple[Tensor, dict[str, Tensor]]
        @return: The main loss and optional a dictionary of sub-losses
            (for logging). Only the main loss is used for
            backpropagation.
        """
        ...

    @cached_property
    def _signature(self) -> dict[str, Parameter]:
        return self._get_signature(self.forward)

    def run(
        self, inputs: Packet[Tensor], labels: Labels
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        """Calls the loss function.

        Validates and prepares the inputs, then calls the loss function.

        @type inputs: Packet[Tensor]
        @param inputs: Outputs from the node.
        @type labels: L{Labels}
        @param labels: Labels from the dataset.
        @rtype: Tensor | tuple[Tensor, dict[str, Tensor]]
        @return: The main loss and optional a dictionary of sub-losses
            (for logging). Only the main loss is used for
            backpropagation.
        @raises IncompatibleError: If the inputs are not compatible with
            the module.
        """
        return self(**self.get_parameters(inputs, labels))
