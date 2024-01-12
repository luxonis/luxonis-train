from abc import ABC
from typing import Generic

from luxonis_ml.utils.registry import AutoRegisterMeta
from pydantic import ValidationError
from torch import Tensor, nn
from typing_extensions import TypeVarTuple, Unpack

from luxonis_train.nodes import BaseNode
from luxonis_train.utils.general import validate_packet
from luxonis_train.utils.types import (
    BaseProtocol,
    IncompatibleException,
    Labels,
    LabelType,
    Packet,
)

Ts = TypeVarTuple("Ts")


class BaseAttachedModule(
    nn.Module, Generic[Unpack[Ts]], ABC, metaclass=AutoRegisterMeta, register=False
):
    """Base class for all modules that are attached to a L{LuxonisNode}.

    Attached modules include losses, metrics and visualizers.

    This class contains a default implementation of `prepare` method, which
    should be sufficient for most simple cases. More complex modules should
    override the `prepare` method.

    @type node: BaseNode
    @ivar node: Reference to the node that this module is attached to.
    @type protocol: type[BaseProtocol]
    @ivar protocol: Schema for validating inputs to the module.
    @type required_labels: list[LabelType]
    @ivar required_labels: List of labels required by this model.
    """

    def __init__(
        self,
        *,
        node: BaseNode | None = None,
        protocol: type[BaseProtocol] | None = None,
        required_labels: list[LabelType] | None = None,
    ):
        """Base class for all modules that are attached to a L{LuxonisNode}.

        @type node: L{BaseNode}
        @param node: Reference to the node that this module is attached to.
        @type protocol: type[BaseProtocol]
        @param protocol: Schema for validating inputs to the module.
        @type required_labels: list[LabelType]
        @param required_labels: List of labels required by this model.
        """
        super().__init__()
        self.required_labels = required_labels or []
        self.protocol = protocol
        self._node = node
        self._epoch = 0

    @property
    def node(self) -> BaseNode:
        """Reference to the node that this module is attached to.

        @type: L{BaseNode}
        @raises RuntimeError: If the node was not provided during initialization.
        """
        if self._node is None:
            raise RuntimeError(
                "Attempt to access `node` reference, but it was not "
                "provided during initialization."
            )
        return self._node

    def prepare(self, inputs: Packet[Tensor], labels: Labels) -> tuple[Unpack[Ts]]:
        """Prepares node outputs for the forward pass of the module.

        This default implementation selects the output and label based on
        C{required_labels} attribute. If not set, then it returns the first
        matching output and label.
        That is the first pair of outputs and labels that have the same type.
        For more complex modules this method should be overridden.

        @type inputs: L{Packet}[Tensor]
        @param inputs: Output from the node, inputs to the attached module.
        @type labels: L{Labels}
        @param labels: Labels from the dataset.

        @rtype: tuple[Unpack[Ts]]
        @return: Prepared inputs. Should allow the following usage with the
            L{forward} method:

                >>> loss.forward(*loss.prepare(outputs, labels))

        @raises NotImplementedError: If the module requires multiple labels.
        @raises IncompatibleException: If the inputs are not compatible with the module.
        """
        if len(self.required_labels) > 1:
            raise NotImplementedError(
                "This module requires multiple labels, the default `prepare` "
                "implementation does not support this."
            )
        if not self.required_labels:
            if "boxes" in inputs and LabelType.BOUNDINGBOX in labels:
                return inputs["boxes"], labels[LabelType.BOUNDINGBOX]  # type: ignore
            if "classes" in inputs and LabelType.CLASSIFICATION in labels:
                return inputs["classes"][0], labels[LabelType.CLASSIFICATION]  # type: ignore
            if "keypoints" in inputs and LabelType.KEYPOINT in labels:
                return inputs["keypoints"], labels[LabelType.KEYPOINT]  # type: ignore
            if "segmentation" in inputs and LabelType.SEGMENTATION in labels:
                return inputs["segmentation"][0], labels[LabelType.SEGMENTATION]  # type: ignore
            raise IncompatibleException(
                f"No matching labels and outputs found for {self.__class__.__name__}"
            )
        label_type = self.required_labels[0]
        return inputs[label_type.value], labels[label_type]  # type: ignore

    def validate(self, inputs: Packet[Tensor], labels: Labels) -> None:
        """Validates that the inputs and labels are compatible with the module.

        @type inputs: L{Packet}[Tensor]
        @param inputs: Output from the node, inputs to the attached module.
        @type labels: L{Labels}
        @param labels: Labels from the dataset. @raises L{IncompatibleException}: If the
            inputs are not compatible with the module.
        """
        for label in self.required_labels:
            if label not in labels:
                raise IncompatibleException.from_missing_label(
                    label, list(labels.keys()), self.__class__.__name__
                )

        if self.protocol is not None:
            try:
                validate_packet(inputs, self.protocol)
            except ValidationError as e:
                raise IncompatibleException.from_validation_error(
                    e, self.__class__.__name__
                ) from e
