"""The base class of every loss, metric, and visualizer.

`BaseAttachedModule` checks that a module fits the task and the type of
its node. `BaseAttachedModule.get_parameters` selects the inputs of a
module from the output packet of the node and the labels, by the
parameter names of the module.

"""

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
from luxonis_ml.utils import AutoRegisterMeta
from torch import Size, Tensor, nn

from luxonis_train.nodes import BaseNode
from luxonis_train.tasks import Metadata, Task
from luxonis_train.typing import Labels, Packet
from luxonis_train.utils import IncompatibleError


class BaseAttachedModule(
    nn.Module, ABC, metaclass=AutoRegisterMeta, register=False
):
    """Base class for all modules that attach to a `BaseNode`.

    `BaseLoss`, `BaseMetric`, and `BaseVisualizer` subclass it. A
    subclass can restrict the nodes that it accepts in two ways:

    - Set the class attribute ``supported_tasks``.
    - Annotate ``node`` in the class body, for example
      ``node: OCRCTCHead``. The constructor then raises
      `IncompatibleError` when the node is not an instance of that
      class. The check reads only the annotations of the nearest class
      that has annotations. Thus the annotations of a subclass hide the
      ``node`` annotation of its parent.

    The properties that read the node, such as `n_classes`, raise
    ``RuntimeError`` when the module has no node.

    Attributes:
        supported_tasks (``Sequence[Task] | None``): The tasks that the
            module supports. The constructor raises `IncompatibleError`
            when the task of the node is not in the sequence. When the
            module gets no task from its node and the sequence holds
            one task, that task becomes the task of the module. ``None``
            accepts a node with any task.

    """

    supported_tasks: Sequence[Task] | None = None

    def __init__(self, *, node: BaseNode | None = None, **kwargs):
        """Initialize the module and select its task.

        The task of the node becomes the task of the module. Without a
        node, or with a node that has no task, the only item of
        ``supported_tasks`` becomes the task. In the other cases, the
        module has no task, and `task` raises ``RuntimeError``.

        Args:
            node (BaseNode | None): The node that the module attaches to.
                The trainer passes it. ``None`` makes the properties that
                read the node raise ``RuntimeError``.
            **kwargs (``Any``): Keyword arguments forwarded to the next
                base class. For a metric, it is the ``torchmetrics``
                ``Metric``. For a loss or a visualizer, it is
                `torch.nn.Module`, which raises ``TypeError`` for any
                keyword argument.

        Raises:
            IncompatibleError: When the task of the node is not in
                ``supported_tasks``, or when the node is not an instance
                of the class in the ``node`` annotation.

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
        """The number of the current training epoch, from ``0``.

        The value comes from `node`. `LuxonisLightningModule` sets it on
        the node at the start of each training epoch.

        Raises:
            RuntimeError: When the module has no node.

        """
        return self.node.current_epoch

    @cached_property
    @abstractmethod
    def _signature(self) -> dict[str, Parameter]: ...

    @property
    def task(self) -> Task:
        """The task of the module.

        The constructor selects it from the node or from
        ``supported_tasks``.

        Raises:
            RuntimeError: When the module has no task.

        """
        if self._task is None:
            raise RuntimeError(
                f"Task of module '{self.name}' is not set. This can happen "
                "if the module does not specify what tasks it supports "
                "or is being connected to a node that also does not "
                "specify its task. Either specify the `task` attribute "
                f"on the node '{self.node.name}', or specify the "
                f"`supported_tasks` attribute on the attached module "
                f"'{self.name}'."
            )
        return self._task

    @property
    def required_labels(self) -> set[str | Metadata]:
        """The labels that the task of the module requires.

        The base implementation returns the `Task.required_labels` of
        `task`. A ``target`` parameter without an underscore selects the
        only label of this set.

        Raises:
            RuntimeError: When the module has no task.

        """
        return self.task.required_labels

    @property
    def name(self) -> str:
        """The class name of the module.

        It is not the alias of the module in the config. The error
        messages of the module use it.

        """
        return self.__class__.__name__

    @property
    def node(self) -> BaseNode:
        """The node that the module attaches to.

        Raises:
            RuntimeError: When the constructor got no node.

        """
        if self._node is None:
            raise RuntimeError(
                "Attempt to access `node` reference, but it was not "
                "provided during initialization."
            )
        return self._node

    @property
    def n_keypoints(self) -> int:
        """The number of keypoints of the node task.

        The value is `BaseNode.n_keypoints` of `node`. It is ``0`` when
        the dataset has no keypoints for the task of the node.

        Raises:
            RuntimeError: When the module has no node, or when the node
                got neither ``n_keypoints`` nor ``dataset_metadata``.

        """
        return self.node.n_keypoints

    @property
    def n_classes(self) -> int:
        """The number of classes of the node task.

        The value is `BaseNode.n_classes` of `node`.

        Raises:
            RuntimeError: When the module has no node, or when the node
                got neither ``n_classes`` nor ``dataset_metadata``.
            ValueError: When the dataset has no task with the
                ``task_name`` of the node.

        """
        return self.node.n_classes

    @property
    def original_in_shape(self) -> Size:
        """The shape of the model input image, ``[C, H, W]``.

        The value is `BaseNode.original_in_shape` of `node`. The shape
        does not include the batch dimension.

        Raises:
            RuntimeError: When the module has no node, or when the node
                got no ``original_in_shape``.

        """
        return self.node.original_in_shape

    @property
    def classes(self) -> bidict[str, int]:
        """The class indices of the node task, keyed by class name.

        The value is `BaseNode.classes` of `node`, a new ``bidict``.

        Raises:
            RuntimeError: When the module has no node, or when the node
                got no ``dataset_metadata``.
            ValueError: When the dataset has no task with the
                ``task_name`` of the node.

        """
        return self.node.classes

    def get_parameters(
        self, predictions: Packet[Tensor], labels: Labels | None = None
    ) -> dict[str, Tensor | list[Tensor] | None]:
        """Select the arguments of the module from a batch.

        The method reads the parameters of `BaseLoss.forward`,
        `BaseMetric.update`, or `BaseVisualizer.forward`, without
        ``self``, ``kwargs``, and the canvases of a visualizer. It picks
        a value for each parameter by its name:

        - A name that starts with ``target`` selects a label. The part
          after the first underscore is the label, so
          ``target_boundingbox`` selects ``<task_name>/boundingbox``. A
          name without an underscore, such as ``target``, selects the
          only label in `required_labels`.
        - A name that starts with ``pred`` selects the packet key
          after the first underscore. A name without an underscore,
          such as ``predictions``, selects the ``main_output`` of
          `task`.
        - Any other name selects the packet key of that name.

        ``<task_name>`` is the ``task_name`` of `node`. The method
        clones each selected tensor, and each tensor of a selected list.
        When a value is missing, a parameter annotated with ``| None``
        gets ``None``. Another parameter with a default value gets no
        entry, so the default applies.

        Args:
            predictions (``Packet[Tensor]``): The output packet of the
                node.
            labels (``Labels | None``): The labels of the batch, keyed
                ``<task_name>/<label>``. ``None`` acts as an empty
                dictionary.

        Returns:
            ``dict[str, Tensor | list[Tensor] | None]``: The values keyed
            by parameter name, ready to pass as keyword arguments.

        Raises:
            RuntimeError: When a parameter without a default value gets
                no value. Also when a ``target`` name has no underscore
                and the task does not require exactly one label. Also
                when a name needs a node or a task that the module does
                not have.
            TypeError: When a value does not match the annotation of its
                parameter.

        Example:
            A missing optional value becomes ``None``. The selected
            tensor is a copy:

            >>> import torch
            >>> from torch import Tensor
            >>> from luxonis_train.attached_modules.losses import BaseLoss
            >>> class Loss(BaseLoss, register=False):
            ...     def forward(
            ...         self, features: Tensor, scale: Tensor | None = None
            ...     ) -> Tensor:
            ...         return features.sum()
            >>> packet = {"features": torch.ones(2)}
            >>> kwargs = Loss().get_parameters(packet)
            >>> kwargs["scale"] is None
            True
            >>> kwargs["features"] is packet["features"]
            False

        """
        kwargs: dict[str, Tensor | list[Tensor] | None] = {}
        labels = labels or {}
        for kwarg_name, parameter in self._signature.items():
            name, data, kind = self._parameter_source(
                kwarg_name, predictions, labels
            )
            self._add_parameter(
                kwargs, name, kwarg_name, data, parameter, kind
            )

        self._validate_parameter_types(kwargs)
        return kwargs

    def _parameter_source(
        self,
        kwarg_name: str,
        predictions: Packet[Tensor],
        labels: Labels,
    ) -> tuple[
        str,
        Mapping[str, list[Tensor] | Tensor],
        Literal["label", "prediction"],
    ]:
        if kwarg_name.startswith("target"):
            return self._target_label_name(kwarg_name), labels, "label"
        return self._prediction_name(kwarg_name), predictions, "prediction"

    def _target_label_name(self, kwarg_name: str) -> str:
        _, *target_name = kwarg_name.split("_", 1)
        if target_name:
            return f"{self.node.task_name}/{target_name[0]}"
        required_labels = self.required_labels
        if len(required_labels) == 1:
            return f"{self.node.task_name}/{next(iter(required_labels))}"
        raise RuntimeError(
            f"Module '{self.name}' is using the wildcard '{kwarg_name}' argument "
            "in the `forward` or `update` signature, but its task "
            f"'{self.task.name}' requires more than one label ({required_labels}). "
            "Unable to determine which label to use. Please specify the labels "
            "using the 'target_{task_type}' pattern "
            f"({[f'target_{label}' for label in required_labels]})."
        )

    def _prediction_name(self, kwarg_name: str) -> str:
        if not kwarg_name.startswith("pred"):
            return kwarg_name
        _, *prediction_name = kwarg_name.split("_", 1)
        return prediction_name[0] if prediction_name else self.task.main_output

    def _add_parameter(
        self,
        kwargs: dict[str, Tensor | list[Tensor] | None],
        name: str,
        kwarg_name: str,
        data: Mapping[str, list[Tensor] | Tensor],
        parameter: Parameter,
        kind: Literal["label", "prediction"],
    ) -> None:
        if name in data:
            value = data[name]
            if isinstance(value, Tensor):
                kwargs[kwarg_name] = value.clone()
            else:
                kwargs[kwarg_name] = [item.clone() for item in value]
            return
        if self._argument_is_optional(parameter):
            kwargs[kwarg_name] = None
            return
        if parameter.default is Parameter.empty:
            source = "dataset" if kind == "label" else "predictions"
            raise RuntimeError(
                f"Module '{self.name}' requires {kind} '{name}', but it is not "
                f"present in the {source}. All available {kind}s: {list(data.keys())}. "
            )

    def _validate_parameter_types(
        self, kwargs: Mapping[str, Tensor | list[Tensor] | None]
    ) -> None:
        for kwarg_name, parameter in self._signature.items():
            if kwarg_name not in kwargs:
                continue
            value = kwargs[kwarg_name]
            if check_type(value, parameter.annotation):
                continue
            raise TypeError(
                f"Module '{self.name}' requires argument '{kwarg_name}' to be "
                f"of type '{parameter.annotation}', but got "
                f"'{type(value).__name__}'."
            )

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
