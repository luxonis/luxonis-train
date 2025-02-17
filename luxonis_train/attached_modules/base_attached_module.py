import inspect
from abc import ABC, abstractmethod
from collections.abc import Callable, Collection, Sequence
from contextlib import suppress
from functools import cached_property
from inspect import Parameter
from types import UnionType
from typing import Union, get_args, get_origin

from bidict import bidict
from luxonis_ml.data.utils import get_task_type
from luxonis_ml.utils.registry import AutoRegisterMeta
from torch import Size, Tensor, nn

from luxonis_train.nodes import BaseNode
from luxonis_train.tasks import Metadata, Task
from luxonis_train.utils import IncompatibleException, Labels, Packet


class BaseAttachedModule(
    nn.Module,
    ABC,
    metaclass=AutoRegisterMeta,
    register=False,
):
    """Base class for all modules that are attached to a L{LuxonisNode}.

    Attached modules include losses, metrics and visualizers.

    This class contains a default implementation of C{prepare} method, which
    should be sufficient for most simple cases. More complex modules should
    override the C{prepare} method.

    When subclassing, the following methods can be overridden:
        - L{prepare}: Prepares node outputs for the forward pass of the module.
          Override this method if the default implementation is not sufficient.

    Additionally, the following attributes can be overridden:
        - L{supported_tasks}: List of task types that the module supports.
          Used to determine which labels to extract from the dataset and to validate
          compatibility with the node based on the node's tasks.

    @type node: BaseNode
    @param node: Reference to the node that this module is attached to.

    @type supported_tasks: list[Task] | None
    @ivar supported_tasks: List of task types that the module supports.
        Elements of the list can be either a single task type or a tuple of
        task types. In case of the latter, the module requires all of the
        specified labels in the tuple to be present.
    """

    supported_tasks: Sequence[Task] | None = None

    def __init__(self, *, node: BaseNode | None = None):
        super().__init__()
        self._node = node
        self._epoch = 0

        if node is not None and node.task is not None:
            if self.supported_tasks is not None:
                if node.task not in self.supported_tasks:
                    raise IncompatibleException(
                        f"Module '{self.name}' is not compatible with the "
                        f"'{node.name}' node. '{self.name}' supports "
                        f" {self.supported_tasks}, but the node's "
                        f"task is '{node.task}'."
                    )
            self._task = node.task

        elif (
            self.supported_tasks is not None and len(self.supported_tasks) == 1
        ):
            self._task = self.supported_tasks[0]

        else:
            self._task = None

        self._check_node_type_override()

    @staticmethod
    def _get_signature(
        func: Callable, exclude: Collection[str] | None = None
    ) -> dict[str, Parameter]:
        exclude = set(exclude or [])
        exclude |= {"self", "kwargs"}
        signature = dict(inspect.signature(func).parameters)
        return {
            name: param
            for name, param in signature.items()
            if name not in exclude
        }

    @cached_property
    @abstractmethod
    def _signature(self) -> dict[str, Parameter]: ...

    @property
    def task(self) -> Task:
        if self._task is None:
            raise RuntimeError(
                f"Task of module '{self.name}' is not set. This can happen "
                "if the module does not specify what tasks it supports "
                "while being connected to a node that also does not "
                "specify its task. Either specify the `task` attribute "
                f"on the node '{self.node.name}', or specify the "
                f"`supported_tasks` atribute on the attached module "
                f"'{self.name}'."
            )
        return self._task

    @property
    def required_labels(self) -> set[str | Metadata]:
        return self.task.required_labels

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
    def classes(self) -> bidict[str, int]:
        """Getter for the class mapping.

        @type: dict[str, int]
        @raises RuntimeError: If the node doesn't define any task.
        @raises ValueError: If the class names are different for
            different tasks. In that case, use the L{get_class_names}
            method.
        """
        return self.node.classes

    def pick_labels(self, labels: Labels) -> Labels:
        required_labels = {
            f"{self.node.task_name}/{label}" for label in self.required_labels
        }
        return {
            get_task_type(label): labels[label]
            for label in required_labels
            if label in labels
        }

    def pick_inputs(
        self, inputs: Packet[Tensor], keys: Collection[str]
    ) -> Packet[Tensor]:
        out = {}
        for expected in keys:
            if expected in inputs:
                out[expected] = inputs[expected]

        missing = set(keys) - set(out.keys())
        if missing:
            raise RuntimeError(
                f"Module '{self.name}' requires the following outputs from the node: "
                f"{list(missing)}, but the node does not provide them. "
                f"All available outputs: {list(inputs.keys())}. "
                "Please make sure you're using the correct node."
            )
        return out

    def _argument_is_optional(self, name: str) -> bool:
        annotation = self._signature[name].annotation
        origin = get_origin(annotation)
        args = get_args(annotation)
        return origin in {Union, UnionType} and type(None) in args

    def get_parameters(
        self, inputs: Packet[Tensor], labels: Labels | None = None
    ) -> Packet[Tensor]:
        input_names = []
        target_names = []
        pred_name = None
        if len(self._signature) == 2:
            pred_name, target_name = self._signature.keys()
            if pred_name in inputs:
                input_names.append(pred_name)
            elif self.task.main_output in inputs:
                input_names.append(self.task.main_output)
            else:
                input_names.append(pred_name)
            target_names.append(target_name)
        else:
            for name in self._signature.keys():
                if name.startswith("target"):
                    target_names.append(name)
                elif name in {"predictions", "prediction", "preds", "pred"}:
                    input_names.append(self.task.main_output)
                    pred_name = name
                elif name in inputs:
                    input_names.append(name)
                else:
                    raise RuntimeError(
                        f"To make use of automatic parameter extraction, the signature of `{self.name}.forward` (or `update` for subclasses of `BaseMetric`) must follow "
                        "one of the following rules: "
                        "1. Exactly two arguments, first one for predictions and second one for targets. "
                        "2. Predictions argument named 'predictions', 'prediction', 'preds', or 'pred' and a target arguments with names starting with 'target'. The predictions argument will be matched to the main output of the node (output named the same as the node's task). "
                        "3. Prediction arguments named the same way as "
                        "keys in the node outputs and target arguments with names starting with 'target'. "
                        f"The node outputs are: {list(inputs.keys())}."
                    )

        predictions = self.pick_inputs(inputs, input_names)
        if pred_name is not None and self.task.main_output in predictions:
            predictions[pred_name] = predictions.pop(self.task.main_output)

        if labels is None:
            targets = {}
            for name in target_names:
                if self._argument_is_optional(name):
                    targets[name] = None
        else:
            all_labels = set(labels.keys())
            labels = self.pick_labels(labels)
            if len(target_names) == len(labels) == 1:
                targets = {target_names[0]: next(iter(labels.values()))}
            else:
                targets = {}
                for name in target_names:
                    label_name = name.replace("target_", "")
                    if label_name not in labels:
                        if self._argument_is_optional(name):
                            targets[name] = None
                        elif self._signature[name].default is Parameter.empty:
                            required = {
                                f"{self.node.task_name}/{name}"
                                for name in self.required_labels
                            }
                            raise RuntimeError(
                                f"Module '{self.name}' requires labels {required}, "
                                f"but some of them are not present in the dataset. "
                                f"All available labels: {all_labels}. "
                            )
                    else:
                        targets[name] = labels[label_name]

        kwargs = predictions | targets

        for key, val in kwargs.items():
            if isinstance(val, Tensor):
                kwargs[key] = val.clone()
            elif isinstance(val, list):
                for i, item in enumerate(val):
                    if isinstance(item, Tensor):
                        val[i] = item.clone()

        for name, param in self._signature.items():
            typ = param.annotation
            if typ == Tensor and not isinstance(kwargs[name], Tensor):
                raise RuntimeError(
                    f"Module '{self.name}' expects a tensor for input '{name}', "
                    f"but the node '{self.node.name}' returned a list. Please make sure "
                    "the node is returning the correct values."
                )

            elif typ == list[Tensor] and not isinstance(kwargs[name], list):
                raise RuntimeError(
                    f"Module '{self.name}' expects a list of tensors for input '{name}', "
                    f"but the node '{self.node.name}' returned a single tensor. Please make sure "
                    "the node is returning the correct values."
                )
        return kwargs

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
