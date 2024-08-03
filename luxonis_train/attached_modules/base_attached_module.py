import logging
from abc import ABC
from typing import Generic

from luxonis_ml.utils.registry import AutoRegisterMeta
from torch import Tensor, nn
from typing_extensions import TypeVarTuple, Unpack

from luxonis_train.nodes import BaseNode
from luxonis_train.utils.types import IncompatibleException, Labels, LabelType, Packet

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
    @type supported_labels: list[LabelType | tuple[LabelType, ...]] | None
    @ivar supported_labels: List of label types that the module supports.
        Elements of the list can be either a single label type or a tuple of
        label types. In case of the latter, the module requires all of the
        specified labels to be present.

        Example:
            1. C{[LabelType.CLASSIFICATION, LabelType.SEGMENTATION]} means that the
                module requires either classification or segmentation labels.
            1. C{[(LabelType.BOUNDINGBOX, LabelType.KEYPOINTS), LabelType.SEGMENTATION]}
                means that the module requires either both bounding box I{and} keypoint
                labels I{or} segmentation labels.
    """

    supported_labels: list[LabelType | tuple[LabelType, ...]] | None = None

    def __init__(self, *, node: BaseNode | None = None):
        """Base class for all modules that are attached to a L{LuxonisNode}.

        @type node: L{BaseNode}
        @param node: Reference to the node that this module is attached to.
        @type protocol: type[BaseProtocol]
        @param protocol: Schema for validating inputs to the module.
        """
        super().__init__()
        self._node = node
        self._epoch = 0

        self._required_labels: tuple[LabelType, ...] | None = None
        if self._node and self.supported_labels and self.node.tasks:
            node_tasks = set(self.node.tasks)
            for required_labels in self.supported_labels:
                if isinstance(required_labels, LabelType):
                    required_labels = (required_labels,)
                if set(required_labels) <= node_tasks:
                    self._required_labels = required_labels
                    break
            else:
                raise ValueError(
                    f"Module {self.name} supports labels {self.supported_labels}, "
                    f"but is connected to node {self.node.name} which does not support any of them. "
                    f"{self.node.name} supports {list(self.node.tasks.keys())}."
                )

    @property
    def name(self) -> str:
        return self.__class__.__name__

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
    def required_labels(self) -> tuple[LabelType, ...]:
        if self._required_labels is None:
            raise ValueError(f"{self.name} does not require any labels.")
        return self._required_labels

    @property
    def node_tasks(self) -> dict[LabelType, str]:
        if self.node.tasks is None:
            raise ValueError("Node must have the `tasks` attribute specified.")
        return self.node.tasks

    def get_label(
        self, labels: Labels, label_type: LabelType | None = None
    ) -> tuple[Tensor, LabelType]:
        """Extracts a specific label from the labels dictionary.

        If the label type is not provided, the first label that matches the
        required label type is returned.

        Example::
            >>> # supported_labels = [LabelType.SEGMENTATION]
            >>> labels = {"segmentation": ..., "boundingbox": ...}
            >>> get_label(labels)
            (..., LabelType.SEGMENTATION)  # returns the first matching label
            >>> get_label(labels, LabelType.BOUNDINGBOX)
            (..., LabelType.BOUNDINGBOX)  # returns the bounding box label
            >>> get_label(labels, LabelType.CLASSIFICATION)
            IncompatibleException: Label 'classification' is missing from the dataset.

        @type labels: L{Labels}
        @param labels: Labels from the dataset.
        @type label_type: LabelType | None
        @param label_type: Type of the label to extract.
        @raises IncompatibleException: If the label is not found in the labels dictionary.
        @raises NotImplementedError: If the module requires multiple labels. For such cases,
            the `prepare` method should be overridden.

        @rtype: tuple[Tensor, LabelType]
        @return: Extracted label and its type.
        """
        if label_type is not None:
            task_name = self.node.get_task_name(label_type)
            if task_name not in labels:
                raise IncompatibleException.from_missing_task(
                    label_type.value, list(labels.keys()), self.name
                )
            return labels[task_name]

        if len(self.required_labels) > 1:
            raise NotImplementedError(
                f"{self.name} requires multiple labels. You must provide the "
                "`label_type` argument to extract the desired label."
            )
        for label, label_type in labels.values():
            if label_type == self.required_labels[0]:
                return label, label_type
        raise IncompatibleException.from_missing_task(
            self.required_labels[0].value, list(labels.keys()), self.name
        )

    def get_input_tensors(
        self, inputs: Packet[Tensor], task_type: LabelType | str | None = None
    ) -> list[Tensor]:
        """Extracts the input tensors from the packet.

        @type inputs: L{Packet}[Tensor]
        @param inputs: Output from the node this module is attached to.
        @type task_type: LabelType | str | None
        @param task_type: Type of the task to extract. Must be provided when the node
            supports multiple tasks or if the module doesn't require any tasks.
        @rtype: list[Tensor]
        @return: Extracted input tensors
        """
        if task_type is not None:
            if isinstance(task_type, LabelType):
                if task_type not in self.node_tasks:
                    raise ValueError(
                        f"Task {task_type.value} is not supported by the node "
                        f"{self.node.name}."
                    )
                return inputs[self.node_tasks[task_type]]
            else:
                if task_type not in inputs:
                    raise ValueError(f"Task {task_type} is not present in the inputs.")
                return inputs[task_type]

        if len(self.required_labels) > 1:
            raise NotImplementedError(
                f"{self.name} requires multiple labels, "
                "you must provide the `task_type` argument to extract the desired input."
            )
        return inputs[self.node_tasks[self.required_labels[0]]]

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
                f"{self.node.name} must have the `tasks` attribute specified "
                f"for {self.name} to make use of the default `prepare` method."
            )
        if self.supported_labels is None:
            raise ValueError(
                f"{self.name} must have the `supported_labels` attribute "
                "specified in order to use the default `prepare` method."
            )
        if len(self.supported_labels) > 1:
            if len(self.node.tasks) > 1:
                raise NotImplementedError(
                    f"{self.name} supports more than one label type"
                    f"and is connected to {self.node.name} node "
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
                        f"Module {self.name} expects a single tensor as input, "
                        f"but got {len(x)} tensors. Using the last tensor. "
                        f"If this is not the desired behavior, please override the "
                        "`prepare` method of the attached module or the `wrap` "
                        f"method of {self.node.name}."
                    )
                    x = x[-1]

        return x, label  # type: ignore
