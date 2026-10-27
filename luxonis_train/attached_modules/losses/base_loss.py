"""The base class every loss inherits."""

from abc import abstractmethod
from functools import cached_property
from inspect import Parameter

import torch
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

    `BaseAttachedModule.get_parameters` describes how `run` fills the
    parameters of `forward` from predictions and labels.

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

        Args:
            final_loss_weight: The factor by which `run` multiplies the
                main value of the loss. The sub-losses stay unscaled.
                The trainer passes the ``weight`` of the loss config
                here.
            **kwargs: Keyword arguments forwarded to
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
            *args: The inputs of the batch. An implementation replaces
                them with named parameters.

        Returns:
            The main value of the loss, or a tuple of the main value and
            a dictionary of sub-losses. The trainer logs the sub-losses
            only when ``trainer.log_sub_losses`` is ``True``. The total
            loss does not include them, so they do not reach the
            gradient.

        """
        ...

    @cached_property
    def _signature(self) -> dict[str, Parameter]:
        return get_signature(self.forward)

    def run(
        self,
        inputs: Packet[Tensor],
        labels: Labels,
        *,
        sample_weights: Tensor | None = None,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        """Resolve the inputs of `forward` and apply the loss weight.

        `BaseAttachedModule.get_parameters` documents how parameter
        names select predictions and labels.

        Args:
            inputs: The output packet of the node.
            labels: The labels of the batch, keyed
                ``<task_name>/<label>``.
            sample_weights: Optional finite nonnegative multipliers of
                shape ``[B]``. The concrete loss applies them before its
                reductions; unsupported losses raise an error.

        Returns:
            The result of `forward`, with ``final_loss_weight`` applied
            to the main value. Sub-losses remain unscaled.

        """
        parameters = self.get_parameters(inputs, labels)
        if sample_weights is not None:
            if "sample_weights" not in self._signature:
                raise ValueError(
                    f"{self.name} does not support sample_weights. "
                    "Use a loss with explicit per-sample weighting support."
                )
            parameters["sample_weights"] = sample_weights
        loss = self(**parameters)
        if isinstance(loss, Tensor):
            return loss * self.__final_loss_weight
        main_loss, sublosses = loss
        return main_loss * self.__final_loss_weight, sublosses

    @staticmethod
    def prepare_sample_weights(
        sample_weights: Tensor | None, reference: Tensor
    ) -> Tensor | None:
        """Validate a batch weight vector and keep its arithmetic in FP32."""
        if sample_weights is None:
            return None
        if sample_weights.ndim != 1 or len(sample_weights) != len(reference):
            raise ValueError("sample_weights must have shape [batch_size]")
        weights = sample_weights.detach().to(
            device=reference.device, dtype=torch.float32
        )
        if not torch.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("sample_weights must be finite and nonnegative")
        return weights
