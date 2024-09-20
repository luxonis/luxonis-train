import logging
from abc import ABC
from contextlib import suppress
from typing import Generic

from luxonis_ml.data import LabelType
from luxonis_ml.utils.registry import AutoRegisterMeta
from torch import Size, Tensor, nn
from typing_extensions import TypeVarTuple, Unpack

from luxonis_train.nodes import BaseNode
from luxonis_train.utils import IncompatibleException, Labels, Packet

logger = logging.getLogger(__name__)

Ts = TypeVarTuple("Ts")


class BaseAttachedModule(
    nn.Module,
    Generic[Unpack[Ts]],
    ABC,
    metaclass=AutoRegisterMeta,
    register=False,
):
    """Base class for all modules that are attached to a L{LuxonisNode}.

    Attached modules include losses, metrics and visualizers.

    This class contains a default implementation of `prepare` method, which
    should be sufficient for most simple cases. More complex modules should
    override the `prepare` method.

    When subclassing, the following methods can be overridden:
        - L{prepare}: Prepares node outputs for the forward pass of the module.
          Override this method if the default implementation is not sufficient.

    Additionally, the following attributes can be overridden:
        - L{supported_labels}: List of label types that the module supports.
          Used to determine which labels to extract from the dataset and to validate
          compatibility with the node based on the node's tasks.

    @type node: BaseNode
    @param node: Reference to the node that this module is attached to.

    @type supported_labels: list[LabelType | tuple[LabelType, ...]] | None
    @ivar supported_labels: List of label types that the module supports.
        Elements of the list can be either a single label type or a tuple of
        label types. In case of the latter, the module requires all of the
        specified labels in the tuple to be present.

        Example:
            - C{[LabelType.CLASSIFICATION, LabelType.SEGMENTATION]} means that the
              module requires either classification or segmentation labels.
            - C{[(LabelType.BOUNDINGBOX, LabelType.KEYPOINTS), LabelType.SEGMENTATION]}
              means that the module requires either both bounding box I{and} keypoint
              labels I{or} segmentation labels.
    """

    supported_labels: list[LabelType | tuple[LabelType, ...]] | None = None

    def __init__(self, *, node: BaseNode | None = None):
        super().__init__()
        self._node = node
        self._epoch = 0

        self.required_labels: list[LabelType] = []
        if self._node and self.supported_labels:
            module_supported = [
                label.value
                if isinstance(label, LabelType)
                else f"({' + '.join(label)})"
                for label in self.supported_labels
            ]
            module_supported = f"[{', '.join(module_supported)}]"
            if not self.node.tasks:
                raise IncompatibleException(
                    f"Module '{self.name}' requires one of the following "
                    f"labels or combinations of labels: {module_supported}, "
                    f"but is connected to node '{self.node.name}' which does not specify any tasks."
                )
            node_tasks = set(self.node.tasks)
            for required_labels in self.supported_labels:
                if isinstance(required_labels, LabelType):
                    required_labels = [required_labels]
                else:
                    required_labels = list(required_labels)
                if set(required_labels) <= node_tasks:
                    self.required_labels = required_labels
                    break
            else:
                node_supported = [task.value for task in self.node.tasks]
                raise IncompatibleException(
                    f"Module '{self.name}' requires one of the following labels or combinations of labels: {module_supported}, "
                    f"but is connected to node '{self.node.name}' which does not support any of them. "
                    f"{self.node.name} supports {node_supported}."
                )
        self._check_node_type_override()

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def node(self) -> BaseNode:
        """Reference to the node that this module is attached to.

        @type: L{BaseNode}
        @raises RuntimeError: If the node was not provided during
            initialization.
        """
        if self._node is None:
            raise RuntimeError(
                "Attempt to access `node` reference, but it was not "
                "provided during initialization."
            )
        return self._node

    @property
    def n_keypoints(self) -> int:
        """Getter for the number of keypoints.

        @type: int
        @raises ValueError: If the node does not support keypoints.
        @raises RuntimeError: If the node doesn't define any task.
        """
        return self.node.n_keypoints

    @property
    def n_classes(self) -> int:
        """Getter for the number of classes.

        @type: int
        @raises RuntimeError: If the node doesn't define any task.
        @raises ValueError: If the number of classes is different for
            different tasks. In that case, use the L{get_n_classes}
            method.
        """
        return self.node.n_classes

    @property
    def original_in_shape(self) -> Size:
        """Getter for the original input shape as [N, H, W].

        @type: Size
        """
        return self.node.original_in_shape

    @property
    def class_names(self) -> list[str]:
        """Getter for the class names.

        @type: list[str]
        @raises RuntimeError: If the node doesn't define any task.
        @raises ValueError: If the class names are different for
            different tasks. In that case, use the L{get_class_names}
            method.
        """
        return self.node.class_names

    @property
    def node_tasks(self) -> dict[LabelType, str]:
        """Getter for the tasks of the attached node.

        @type: dict[LabelType, str]
        @raises RuntimeError: If the node does not have the `tasks` attribute set.
        """
        if self.node._tasks is None:
            raise RuntimeError(
                "Node must have the `tasks` attribute specified."
            )
        return self.node._tasks

    def get_label(
        self, labels: Labels, label_type: LabelType | None = None
    ) -> Tensor:
        """Extracts a specific label from the labels dictionary.

        If the label type is not provided, the first label that matches the
        required label type is returned.

        Example::
            >>> # supported_labels = [LabelType.SEGMENTATION]
            >>> labels = {"segmentation": seg_tensor, "boundingbox": bbox_tensor}
            >>> get_label(labels)
            seg_tensor  # returns the first matching label
            >>> get_label(labels, LabelType.BOUNDINGBOX)
            bbox_tensor # returns the bounding box label
            >>> get_label(labels, LabelType.CLASSIFICATION)
            IncompatibleException: Label 'classification' is missing from the dataset.

        @type labels: L{Labels}
        @param labels: Labels from the dataset.
        @type label_type: LabelType | None
        @param label_type: Type of the label to extract.

        @rtype: Tensor
        @return: Extracted label

        @raises ValueError: If the module requires multiple labels and the C{label_type} is not provided.
        @raises IncompatibleException: If the label is not found in the labels dictionary.
        """
        return self._get_label(labels, label_type)[0]

    def _get_label(
        self, labels: Labels, label_type: LabelType | None = None
    ) -> tuple[Tensor, LabelType]:
        if label_type is None:
            if len(self.required_labels) == 1:
                label_type = self.required_labels[0]

        if label_type is not None:
            task_name = self.node.get_task_name(label_type)
            if task_name not in labels:
                raise IncompatibleException.from_missing_task(
                    label_type.value, list(labels.keys()), self.name
                )
            return labels[task_name]

        raise ValueError(
            f"{self.name} requires multiple labels. You must provide the "
            "`label_type` argument to extract the desired label."
        )

    def get_input_tensors(
        self, inputs: Packet[Tensor], task_type: LabelType | str | None = None
    ) -> list[Tensor]:
        """Extracts the input tensors from the packet.

        Example::
            >>> # supported_labels = [LabelType.SEGMENTATION]
            >>> # node.tasks = {LabelType.SEGMENTATION: "segmentation-task"}
            >>> inputs = [{"segmentation-task": [seg_tensor]}, {"features": [feat_tensor]}]
            >>> get_input_tensors(inputs)  # matches supported labels to node's tasks
            [seg_tensor]
            >>> get_input_tensors(inputs, "features")
            [feat_tensor]
            >>> get_input_tensors(inputs, LabelType.CLASSIFICATION)
            ValueError: Task 'classification' is not supported by the node.

        @type inputs: L{Packet}[Tensor]
        @param inputs: Output from the node this module is attached to.
        @type task_type: LabelType | str | None
        @param task_type: Type of the task to extract. Must be provided when the node
            supports multiple tasks or if the module doesn't require any tasks.
        @rtype: list[Tensor]
        @return: Extracted input tensors

        @raises IncompatibleException: If the task type is not supported by the node.
        @raises IncompatibleException: If the task is not present in the inputs.

        @raises ValueError: If the module requires multiple labels.
            For such cases, the `prepare` method should be overridden.
        """
        if task_type is not None:
            if isinstance(task_type, LabelType):
                if task_type not in self.node_tasks:
                    raise IncompatibleException(
                        f"Task {task_type.value} is not supported by the node "
                        f"{self.node.name}."
                    )
                return inputs[self.node_tasks[task_type]]
            else:
                if task_type not in inputs:
                    raise IncompatibleException(
                        f"Task {task_type} is not present in the inputs."
                    )
                return inputs[task_type]

        if len(self.required_labels) > 1:
            raise ValueError(
                f"{self.name} requires multiple labels, "
                "you must provide the `task_type` argument to extract the desired input."
            )
        return inputs[self.node_tasks[self.required_labels[0]]]

    def prepare(
        self, inputs: Packet[Tensor], labels: Labels | None
    ) -> tuple[Unpack[Ts]]:
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
            L{forward} method::

                >>> loss.forward(*loss.prepare(outputs, labels))

        @raises RuntimeError: If the module requires multiple labels and
            is connected to a multi-task node. In this case, the default
            implementation cannot be used and the C{prepare} method should be overridden.

        @raises RuntimeError: If the C{tasks} attribute is not set on the node.
        @raises RuntimeError: If the C{supported_labels} attribute is not set on the module.
        """
        if self.node._tasks is None:
            raise RuntimeError(
                f"{self.node.name} must have the `tasks` attribute specified "
                f"for {self.name} to make use of the default `prepare` method."
            )
        if self.supported_labels is None:
            raise RuntimeError(
                f"{self.name} must have the `supported_labels` attribute "
                "specified in order to use the default `prepare` method."
            )
        if len(self.supported_labels) > 1:
            if len(self.node_tasks) > 1:
                raise RuntimeError(
                    f"{self.name} supports more than one label type"
                    f"and is connected to {self.node.name} node "
                    "which is a multi-task node. The default `prepare` "
                    "implementation cannot be used in this case."
                )
            self.supported_labels = list(
                set(self.supported_labels) & set(self.node_tasks)
            )
        x = self.get_input_tensors(inputs)
        # NOTE: Check the logic below, if x needs to be modified withoud fulfilling the condition
        if labels is not None:
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
        else:
            label = None

        return x, label  # type: ignore

    def _check_node_type_override(self) -> None:
        if "node" not in self.__annotations__:
            return

        node_type = self.__annotations__["node"]
        with suppress(RuntimeError):
            if not isinstance(self.node, node_type):
                raise IncompatibleException(
                    f"Module '{self.name}' is attached to the '{self.node.name}' node, "
                    f"but '{self.name}' is only compatible with nodes of type '{node_type.__name__}'."
                )
