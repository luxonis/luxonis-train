"""The base class every loss inherits."""

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
    """Base class for all losses.

    Every subclass registers itself in the `LOSSES` registry under its
    class name, unless its class statement passes ``register=False``. A
    ``register_name`` in the class statement replaces the class name. A
    config names a registered loss in the ``losses`` list of a node. A
    subclass implements `forward`.

    `run` fills the parameters of `forward` by their names:

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

    The trainer calls `run` on each training, validation, and test
    batch. It sums the main values of all losses into the total loss.
    The training step backpropagates only this total.

    Example:
        The example loss reads the ``features`` key of the node output.
        The main value is the mean, and the sub-loss ``"max"`` is the
        maximum. `run` multiplies only the main value by
        ``final_loss_weight``. ``register=False`` keeps the class out of
        the `LOSSES` registry.

        >>> import torch
        >>> from torch import Tensor
        >>> class MeanLoss(BaseLoss, register=False):
        ...     def forward(
        ...         self, features: Tensor
        ...     ) -> tuple[Tensor, dict[str, Tensor]]:
        ...         return features.mean(), {"max": features.max()}
        >>> loss = MeanLoss(final_loss_weight=2.0)
        >>> packet = {"features": torch.tensor([1.0, 3.0])}
        >>> main, sub_losses = loss.run(packet, {})
        >>> main.item(), sub_losses["max"].item()
        (4.0, 3.0)

    """

    @typechecked
    def __init__(self, final_loss_weight: float = 1.0, **kwargs):
        """Initialize the loss and store the factor of its main value.

        The ``typechecked`` decorator of ``typeguard`` raises
        ``TypeCheckError`` when ``final_loss_weight`` is not a ``float``
        or an ``int``. `BaseAttachedModule` raises `IncompatibleError`
        when the node does not fit the loss.

        Args:
            final_loss_weight (float): The factor by which `run`
                multiplies the main value of the loss. The sub-losses
                stay unscaled. The trainer passes the ``weight`` of the
                loss config here.
            **kwargs (``Any``): Keyword arguments forwarded to
                `BaseAttachedModule`, such as ``node``.

        """
        super().__init__(**kwargs)
        self.__final_loss_weight = final_loss_weight

    @abstractmethod
    def forward(
        self, *args: Tensor | list[Tensor]
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        """Compute the loss for one batch.

        An implementation declares one named parameter for each input.
        `run` fills the parameters by name, as the class docstring
        describes, and passes them as keyword arguments. The tensors
        are copies, so a change in place does not reach the node output
        or the labels.

        Args:
            *args (``Tensor | list[Tensor]``): The inputs of the batch.
                An implementation replaces them with named parameters.

        Returns:
            ``Tensor | tuple[Tensor, dict[str, Tensor]]``: The main value
            of the loss, or a tuple of the main value and a dictionary
            of sub-losses. The trainer logs the sub-losses only when
            ``trainer.log_sub_losses`` is ``True``. The total loss does
            not include them, so they do not reach the gradient.

        """
        ...

    @cached_property
    def _signature(self) -> dict[str, Parameter]:
        """The parameters of `forward` that `run` fills.

        `get_signature` leaves out ``self`` and ``kwargs``.

        """
        return get_signature(self.forward)

    def run(
        self, inputs: Packet[Tensor], labels: Labels
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        """Select the inputs of `forward` from a batch and run the loss.

        `BaseAttachedModule.get_parameters` picks a value for each
        parameter of `forward` by its name, as the class docstring
        describes. It clones every tensor that it picks. The method
        calls the module with these values, which runs `forward`. It
        then multiplies the main value by ``final_loss_weight``.

        When a value is missing, a parameter annotated with ``| None``
        receives ``None``, even when it has a default value. Another
        parameter with a default value keeps the default. Any other
        parameter raises ``RuntimeError``. A ``target`` name without an
        underscore also raises ``RuntimeError`` when the task requires
        more than one label. A ``pred`` or ``target`` name without an
        underscore raises ``RuntimeError`` when the module has no task.
        A ``target`` name raises ``RuntimeError`` when the module has no
        node. A value that does not match the annotation of its
        parameter raises ``TypeError``.

        Args:
            inputs (``Packet[Tensor]``): The output packet of the node.
            labels (``Labels``): The labels of the batch, keyed
                ``<task_name>/<label>``.

        Returns:
            ``Tensor | tuple[Tensor, dict[str, Tensor]]``: The result of
            `forward` in the same form. The method multiplies the main
            value by ``final_loss_weight`` and keeps the sub-losses as
            they are.

        """
        loss = self(**self.get_parameters(inputs, labels))
        if isinstance(loss, Tensor):
            return loss * self.__final_loss_weight
        main_loss, sublosses = loss
        return main_loss * self.__final_loss_weight, sublosses
