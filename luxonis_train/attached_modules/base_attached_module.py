import logging
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

logger = logging.getLogger(__name__)

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
    """

    supported_labels: list[LabelType | tuple[LabelType, ...]] | None = None

    def __init__(
        self,
        *,
        node: BaseNode | None = None,
        protocol: type[BaseProtocol] | None = None,
    ):
        """Base class for all modules that are attached to a L{LuxonisNode}.

        @type node: L{BaseNode}
        @param node: Reference to the node that this module is attached to.
        @type protocol: type[BaseProtocol]
        @param protocol: Schema for validating inputs to the module.
        """
        super().__init__()
        self.protocol = protocol
        self._node = node
        self._epoch = 0

        self._labels = None
        if self._node and self.supported_labels and self.node.tasks:
            node_tasks = set(self.node.tasks)
            for supported_labels in self.supported_labels:
                if isinstance(supported_labels, LabelType):
                    supported_labels = (supported_labels,)
                if set(supported_labels) <= node_tasks:
                    self._labels = supported_labels
                    break
            else:
                raise ValueError(
                    f"Module {self.module_name} supports labels {self.supported_labels}, "
                    f"but is connected to node {self.node.node_name} which does not support any of them. "
                    f"{self.node.node_name} supports {list(self.node.tasks.keys())}."
                )
        # print(f"{self.module_name}, {self.supported_labels}, {self.node.tasks}")
        # print(f"{self.module_name}, {self._labels}")

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

    @property
    def labels_dict(self) -> tuple[LabelType, ...]:
        if self._labels is None:
            raise ValueError(f"{self.module_name} does not require any labels.")
        return self._labels

    @property
    def node_tasks(self) -> dict[LabelType, str]:
        if self.node.tasks is None:
            raise ValueError("Node must have the `tasks` attribute specified.")
        return self.node.tasks

    def get_label(self, labels: Labels) -> tuple[Tensor, LabelType]:
        if len(self.labels_dict) > 1:
            raise NotImplementedError(
                f"{self.module_name} requires multiple labels, "
                "the default `prepare` implementation does not support this."
            )
        for label, label_type in labels.values():
            if label_type == self.labels_dict[0]:
                return label, label_type
        raise IncompatibleException.from_missing_task(
            self.labels_dict[0].value, list(labels.keys()), self.module_name
        )

    def get_input_tensors(self, inputs: Packet[Tensor]) -> list[Tensor]:
        if self.protocol is not None:
            return inputs[self.protocol.get_task()]
        return inputs[self.node_tasks[self.labels_dict[0]]]

    def prepare(self, inputs: Packet[Tensor], labels: Labels) -> tuple[Unpack[Ts]]:
        """Prepares node outputs for the forward pass of the module.

        This default implementation selects the output and label based on
        C{supported_labels} attribute. If not set, then it returns the first
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
        if self.node.tasks is None:
            raise ValueError(
                f"{self.node.node_name} must have the `tasks` attribute specified "
                f"for {self.module_name} to make use of the default `prepare` method."
            )
        if self.supported_labels is None:
            raise ValueError(
                f"{self.module_name} must have the `supported_labels` attribute "
                "specified in order to use the default `prepare` method."
            )
        if len(self.supported_labels) > 1:
            if len(self.node.tasks) > 1:
                raise NotImplementedError(
                    f"{self.module_name} supports more than one label type"
                    f"and is connected to {self.node.node_name} node "
                    "which is a multi-task node. The default `prepare` "
                    "implementation cannot be used in this case."
                )
            self.supported_labels = list(
                set(self.supported_labels) & set(self.node.tasks)
            )
        x = self.get_input_tensors(inputs)
        label, label_type = self.get_label(labels)
        if label_type in [LabelType.CLASSIFICATION, LabelType.SEGMENTATION]:
            if isinstance(x, list):
                if len(x) == 1:
                    x = x[0]
                else:
                    logger.warning(
                        f"Module {self.module_name} expects a single tensor as input, "
                        f"but got {len(x)} tensors. Using the last tensor. "
                        f"If this is not the desired behavior, please override the "
                        "`prepare` method of the attached module or the `wrap` "
                        f"method of {self.node.node_name}."
                    )
                    x = x[-1]

        return x, label  # type: ignore

    @property
    def module_name(self) -> str:
        return self.__class__.__name__

    def validate(self, inputs: Packet[Tensor], labels: Labels) -> None:
        """Validates that the inputs and labels are compatible with the module.

        @type inputs: L{Packet}[Tensor]
        @param inputs: Output from the node, inputs to the attached module.
        @type labels: L{Labels}
        @param labels: Labels from the dataset. @raises L{IncompatibleException}: If the
            inputs are not compatible with the module.
        """
        if self.node.tasks is not None:
            for task in self.node.tasks.values():
                if task not in labels:
                    raise IncompatibleException.from_missing_task(
                        task, list(labels.keys()), self.module_name
                    )

        if self.protocol is not None:
            try:
                validate_packet(inputs, self.protocol)
            except ValidationError as e:
                raise IncompatibleException.from_validation_error(
                    e, self.module_name
                ) from e
