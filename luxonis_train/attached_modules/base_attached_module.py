from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from contextlib import suppress
from functools import cached_property
from inspect import Parameter
from types import UnionType
from typing import Literal, Union, get_args, get_origin

from bidict import bidict
from loguru import logger
from luxonis_ml.typing import check_type
from luxonis_ml.utils.registry import AutoRegisterMeta
from torch import Size, Tensor, nn

from luxonis_train.nodes import BaseNode
from luxonis_train.tasks import Metadata, Task
from luxonis_train.typing import Labels, Packet
from luxonis_train.utils import IncompatibleError


class BaseAttachedModule(
    nn.Module, ABC, metaclass=AutoRegisterMeta, register=False
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


    @type supported_tasks: list[Task] | None
    @ivar supported_tasks: List of task types that the module supports.
        Elements of the list can be either a single task type or a tuple of
        task types. In case of the latter, the module requires all of the
        specified labels in the tuple to be present.
    """

    supported_tasks: Sequence[Task] | None = None

    def __init__(self, *, node: BaseNode | None = None, **kwargs):
        """Constructor for teh C{BaseAttachedModule}

        @type node: BaseNode
        @param node: Reference to the node that this module is attached
            to.
        @param kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self._node = node

        if node is not None and node.task is not None:
            if (
                self.supported_tasks is not None
                and node.task not in self.supported_tasks
            ):
                raise IncompatibleError(
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

    @property
    def current_epoch(self) -> int:
        return self.node.current_epoch

    @cached_property
    @abstractmethod
    def _signature(self) -> dict[str, Parameter]: ...

    @property
    def task(self) -> Task:
        if self._task is None:
            raise RuntimeError(
                f"Task of module '{self.name}' is not set. This can happen "
                "if the module does not specify what tasks it supports "
                "or is being connected to a node that also does not "
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

    def get_parameters(
        self, predictions: Packet[Tensor], labels: Labels | None = None
    ) -> dict[str, Tensor | list[Tensor]]:
        kwargs = {}
        labels = labels or {}

        def _add_to_kwargs(
            name: str,
            kwarg_name: str,
            data: Mapping[str, list[Tensor] | Tensor],
            parameter: Parameter,
            kind: Literal["label", "prediction"],
        ) -> None:
            if name not in data:
                if self._argument_is_optional(parameter):
                    kwargs[kwarg_name] = None
                elif parameter.default is Parameter.empty:
                    raise RuntimeError(
                        f"Module '{self.name}' requires {kind} '{name}', "
                        f"but it is not present in the "
                        f"{'dataset' if kind == 'label' else 'predictions'}. "
                        f"All available {kind}s: {list(data.keys())}. "
                    )
            else:
                val = data[name]
                if isinstance(val, Tensor):
                    kwargs[kwarg_name] = val.clone()
                elif isinstance(val, list):
                    kwargs[kwarg_name] = [v.clone() for v in val]
                else:
                    kwargs[kwarg_name] = val

        for kwarg_name, parameter in self._signature.items():
            if kwarg_name.startswith("target"):
                _, *target_name = kwarg_name.split("_", 1)
                if target_name:
                    label_name = f"{self.node.task_name}/{target_name[0]}"
                else:
                    required_labels = self.required_labels
                    if len(required_labels) == 1:
                        label_name = f"{self.node.task_name}/{next(iter(required_labels))}"
                    else:
                        raise RuntimeError(
                            f"Module '{self.name}' is using the wildcard '{kwarg_name}' "
                            f"argument in the `forward` or `update` signature, "
                            f"but its task '{self.task.name}' requires more than one label "
                            f"({self.required_labels}). "
                            "Unable to determine which of the labels to use. Please specify "
                            "the labels using the 'target_{task_type}' pattern "
                            f"({[f'target_{label}' for label in self.required_labels]})."
                        )
                _add_to_kwargs(
                    label_name, kwarg_name, labels, parameter, "label"
                )
            else:
                if kwarg_name.startswith("pred"):
                    _, *prediction_name = kwarg_name.split("_", 1)
                    if prediction_name:
                        prediction_name = prediction_name[0]
                    else:
                        prediction_name = self.task.main_output
                else:
                    prediction_name = kwarg_name
                _add_to_kwargs(
                    prediction_name,
                    kwarg_name,
                    predictions,
                    parameter,
                    "prediction",
                )

        for kwarg_name, parameter in self._signature.items():
            if not check_type(kwargs[kwarg_name], parameter.annotation):
                raise TypeError(
                    f"Module '{self.name}' requires argument '{kwarg_name}' "
                    f"to be of type '{parameter.annotation}', but got "
                    f"'{type(kwargs[kwarg_name]).__name__}'."
                )

        return kwargs

    def _check_node_type_override(self) -> None:
        if "node" not in self.__annotations__:
            return

        node_type = self.__annotations__["node"]
        with suppress(RuntimeError):
            if not isinstance(self.node, node_type):
                raise IncompatibleError(
                    f"Module '{self.name}' is attached to the '{self.node.name}' node, "
                    f"but '{self.name}' is only compatible with nodes of type '{node_type.__name__}'."
                )

    def _argument_is_optional(self, parameter: Parameter) -> bool:
        annotation = parameter.annotation
        origin = get_origin(annotation)
        args = get_args(annotation)
        return origin in {Union, UnionType} and type(None) in args

    def _infer_torchmetrics_task(self, **kwargs) -> str:
        task = kwargs.get("task")
        if task is None:
            if "num_classes" in kwargs:
                task = "binary" if kwargs["num_classes"] == 1 else "multiclass"
            elif "num_labels" in kwargs:
                task = "multilabel"
            else:
                with suppress(RuntimeError, ValueError):
                    task = "binary" if self.n_classes == 1 else "multiclass"
            if task is not None:
                logger.warning(
                    "Parameter 'task' was not specified for `TorchMetric` "
                    f"based '{self.name}'. Assuming task type '{task}' "
                    "based on the number of classes. "
                    "If this is incorrect, please specify the "
                    "'task' parameter in the config."
                )

        if task is None:
            raise ValueError(
                f"'{self.name}' does not have the 'task' parameter set. "
                "and it is not possible to infer it from the other arguments. "
                "You can either set the 'task' parameter explicitly, "
                "provide either 'num_classes' or 'num_labels' argument, "
                "or use this metric with a node. "
                "The 'task' can be one of 'binary', 'multiclass', "
                "or 'multilabel'. "
            )
        if task not in {"binary", "multiclass", "multilabel"}:
            raise ValueError(
                f"Invalid task type '{task}' for '{self.name}'. "
                "The 'task' can be one of 'binary', 'multiclass', "
                "or 'multilabel'."
            )
        return task
