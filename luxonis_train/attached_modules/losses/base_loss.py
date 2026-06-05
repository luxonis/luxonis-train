from abc import abstractmethod
from functools import cached_property
from inspect import Parameter

from torch import Tensor
from typeguard import typechecked

from luxonis_train.attached_modules import BaseAttachedModule
from luxonis_train.registry import LOSSES
from luxonis_train.typing import Labels, Packet
from luxonis_train.utils import get_signature


class BaseLoss(BaseAttachedModule, register=False, registry=LOSSES):
    """A base class for all loss functions.

    This class defines the basic interface for all loss functions. It
    utilizes automatic registration of defined subclasses to a `LOSSES`
    registry.

    """

    @typechecked
    def __init__(self, final_loss_weight: float = 1.0, **kwargs):
        """Initialize the base loss.

        Args:
            final_loss_weight (float): Optional weight by which the final loss is multiplied.
            **kwargs (``Any``): Keyword arguments forwarded to the parent class.

        """
        super().__init__(**kwargs)
        self.__final_loss_weight = final_loss_weight

    @abstractmethod
    def forward(
        self, *args: Tensor | list[Tensor]
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        """Forward pass of the loss function.

        Returns:
            ``Tensor | tuple[Tensor, dict[str, Tensor]]``: The main loss and optional a dictionary of sub-losses (for logging). Only the main loss is used for backpropagation.

        """
        ...

    @cached_property
    def _signature(self) -> dict[str, Parameter]:
        return get_signature(self.forward)

    def run(
        self, inputs: Packet[Tensor], labels: Labels
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        """Call the loss function after validating and preparing the
        inputs.

        Args:
            inputs (``Packet[Tensor]``): Outputs from the node.
            labels (`Labels`): Labels from the dataset.

        Returns:
            ``Tensor | tuple[Tensor, dict[str, Tensor]]``: The main loss and optional a dictionary of sub-losses (for logging). Only the main loss is used for backpropagation.

        Raises:
            IncompatibleError: If the inputs are not compatible with the module.

        """
        loss = self(**self.get_parameters(inputs, labels))
        if isinstance(loss, Tensor):
            return loss * self.__final_loss_weight
        main_loss, sublosses = loss
        return main_loss * self.__final_loss_weight, sublosses
