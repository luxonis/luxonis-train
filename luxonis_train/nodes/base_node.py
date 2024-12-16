import inspect
import logging
from abc import ABC, abstractmethod
from contextlib import suppress
from typing import Generic, TypeVar

import torch
from luxonis_ml.utils.registry import AutoRegisterMeta
from torch import Size, Tensor, nn
from typeguard import TypeCheckError, check_type

from luxonis_train.enums import TaskType
from luxonis_train.utils import (
    AttachIndexType,
    DatasetMetadata,
    IncompatibleException,
    Packet,
    safe_download,
)
from luxonis_train.utils.registry import NODES

ForwardOutputT = TypeVar("ForwardOutputT")
ForwardInputT = TypeVar("ForwardInputT")

logger = logging.getLogger(__name__)


class BaseNode(
    nn.Module,
    ABC,
    Generic[ForwardInputT, ForwardOutputT],
    metaclass=AutoRegisterMeta,
    register=False,
    registry=NODES,
):
    """A base class for all model nodes.

    This class defines the basic interface for all nodes.

    Furthermore, it utilizes automatic registration of defined subclasses
    to a L{NODES} registry.

    Inputs and outputs of nodes are defined as L{Packet}s. A L{Packet} is a dictionary
    of lists of tensors. Each key in the dictionary represents a different output
    from the previous node. Input to the node is a list of L{Packet}s, output is a single L{Packet}.

    When the node is called, the inputs are sent to the L{unwrap} method.
    The C{unwrap} method should return a valid input to the L{forward} method.
    Outputs of the C{forward} method are then sent to L{wrap} method,
    which wraps the output into a C{Packet}. The wrapped C{Packet} is the final output of the node.

    The L{run} method combines the C{unwrap}, C{forward} and C{wrap} methods
    together with input validation.

    When subclassing, the following methods should be implemented:
        - L{forward}: Forward pass of the module.
        - L{unwrap}: Optional. Unwraps the inputs from the input packet.
            The default implementation expects a single input with C{features} key.
        - L{wrap}: Optional. Wraps the output of the forward pass
            into a C{Packet[Tensor]}. The default implementation expects wraps the output
            of the forward pass into a packet with either "features" or the task name as the key.

    Additionally, the following class attributes can be defined:
        - L{attach_index}: Index of previous output that this node attaches to.
        - L{tasks}: Dictionary of tasks that the node supports.

    Example::
        class MyNode(BaseNode):
            # equivalent to C{tasks = {TaskType.CLASSIFICATION: "classification"}}
            tasks = [TaskType.CLASSIFICATION]

            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.nn = nn.Sequential(
                    nn.Linear(10, 10),
                    nn.ReLU(),
                    nn.Linear(10, 10),
                )

            # Roughly equivalent to the default implementation
            def unwrap(self, inputs: list[Packet[Tensor]]) -> Tensor:
                assert len(inputs) == 1
                assert "features" in inputs[0]
                return inputs[0]["features"]

            def forward(self, inputs: Tensor) -> Tensor:
                return self.nn(inputs)

            # Roughly equivalent to the default implementation
            def wrap(output: Tensor) -> Packet[Tensor]:
                # The key of the main node output have to be the same as the
                # default task name for it to be automatically recognized
                # by the attached modules.
                return {"classification": [output]}


    @type attach_index: AttachIndexType
    @ivar attach_index: Index of previous output that this node attaches to.
        Can be a single integer to specify a single output, a tuple of
        two or three integers to specify a range of outputs or C{"all"} to
        specify all outputs. Defaults to "all". Python indexing conventions apply.

    @type tasks: list[TaskType] | dict[TaskType, str] | None
    @ivar tasks: Dictionary of tasks that the node supports. Should be defined
        by the user as a class attribute. The key is the task type and the value
        is the name of the task. For example:
        C{{TaskType.CLASSIFICATION: "classification"}}.
        Only needs to be defined for head nodes.
    """

    attach_index: AttachIndexType
    tasks: list[TaskType] | dict[TaskType, str] | None = None

    def __init__(
        self,
        *,
        input_shapes: list[Packet[Size]] | None = None,
        original_in_shape: Size | None = None,
        dataset_metadata: DatasetMetadata | None = None,
        n_classes: int | None = None,
        n_keypoints: int | None = None,
        in_sizes: Size | list[Size] | None = None,
        remove_on_export: bool = False,
        export_output_names: list[str] | None = None,
        attach_index: AttachIndexType | None = None,
        _tasks: dict[TaskType, str] | None = None,
    ):
        """Constructor for the C{BaseNode}.

        @type input_shapes: list[Packet[Size]] | None
        @param input_shapes: List of input shapes for the module.
        @type original_in_shape: Size | None
        @param original_in_shape: Original input shape of the model.
            Some nodes won't function if not provided.
        @type dataset_metadata: L{DatasetMetadata} | None
        @param dataset_metadata: Metadata of the dataset. Some nodes
            won't function if not provided.
        @type n_classes: int | None
        @param n_classes: Number of classes in the dataset. Provide only
            in case C{dataset_metadata} is not provided. Defaults to
            None.
        @type in_sizes: Size | list[Size] | None
        @param in_sizes: List of input sizes for the node. Provide only
            in case the C{input_shapes} were not provided.
        @type remove_on_export: bool
        @param remove_on_export: If set to True, the node will be removed
            from the model during export. Defaults to False.
        @type export_output_names: list[str] | None
        @param export_output_names: List of output names for the export.
        @type attach_index: AttachIndexType
        @param attach_index: Index of previous output that this node
            attaches to. Can be a single integer to specify a single
            output, a tuple of two or three integers to specify a range
            of outputs or C{"all"} to specify all outputs. Defaults to
            "all". Python indexing conventions apply. If provided as a
            constructor argument, overrides the class attribute.
        @type _tasks: dict[TaskType, str] | None
        @param _tasks: Dictionary of tasks that the node supports.
            Overrides the class L{tasks} attribute. Shouldn't be
            provided by the user in most cases.
        """
        super().__init__()

        if attach_index is not None:
            logger.warning(
                f"Node {self.name} overrides `attach_index` "
                f"by setting it to '{attach_index}'. "
                "Make sure this is intended."
            )
            self.attach_index = attach_index
        self._tasks = None
        if _tasks is not None:
            self._tasks = _tasks
        elif self.tasks is not None:
            self._tasks = self._process_tasks(self.tasks)

        if getattr(self, "attach_index", None) is None:
            parameters = inspect.signature(self.forward).parameters
            inputs_forward_type = parameters.get(
                "inputs", parameters.get("input", parameters.get("x", None))
            )
            if (
                inputs_forward_type is not None
                and inputs_forward_type.annotation == Tensor
            ):
                self.attach_index = -1
            else:
                self.attach_index = "all"

        self._input_shapes = input_shapes
        self._original_in_shape = original_in_shape
        self._dataset_metadata = dataset_metadata
        self._n_classes = n_classes
        self._n_keypoints = n_keypoints
        self._export = False
        self._remove_on_export = remove_on_export
        self._export_output_names = export_output_names
        self._epoch = 0
        self._in_sizes = in_sizes

        self._check_type_overrides()

    @staticmethod
    def _process_tasks(
        tasks: dict[TaskType, str] | list[TaskType],
    ) -> dict[TaskType, str]:
        if isinstance(tasks, dict):
            return tasks
        else:
            return {task: task.value for task in tasks}

    def _check_type_overrides(self) -> None:
        properties = []
        for name, value in inspect.getmembers(self.__class__):
            if isinstance(value, property):
                properties.append(name)
        for name, typ in self.__annotations__.items():
            if name in properties:
                with suppress(RuntimeError):
                    value = getattr(self, name)
                    try:
                        check_type(value, typ)
                    except TypeCheckError as e:
                        raise IncompatibleException(
                            f"Node '{self.name}' specifies the type of the property `{name}` as `{typ}`, "
                            f"but received `{type(value)}`. "
                            f"This may indicate that the '{self.name}' node is "
                            "not compatible with its predecessor."
                        ) from e

    def get_task_name(self, task: TaskType) -> str:
        """Gets the name of a task for a particular C{TaskType}.

        @type task: TaskType
        @param task: Task to get the name for.
        @rtype: str
        @return: Name of the task.
        @raises RuntimeError: If the node does not define any tasks.
        @raises ValueError: If the task is not supported by the node.
        """
        if not self._tasks:
            raise RuntimeError(f"Node '{self.name}' does not define any task.")

        if task not in self._tasks:
            raise ValueError(
                f"Node '{self.name}' does not support the '{task.value}' task."
            )
        return self._tasks[task]

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def task(self) -> str:
        """Getter for the task.

        @type: str
        @raises RuntimeError: If the node doesn't define any task.
        @raises ValueError: If the node defines more than one task. In
            that case, use the L{get_task_name} method instead.
        """
        if not self._tasks:
            raise RuntimeError(f"{self.name} does not define any task.")

        if len(self._tasks) > 1:
            raise ValueError(
                f"Node {self.name} has multiple tasks defined. "
                "Use the `get_task_name` method instead."
            )
        return next(iter(self._tasks.values()))

    def get_n_classes(self, task: TaskType) -> int:
        """Gets the number of classes for a particular task.

        @type task: TaskType
        @param task: Task to get the number of classes for.
        @rtype: int
        @return: Number of classes for the task.
        """
        return self.dataset_metadata.n_classes(self.get_task_name(task))

    def get_class_names(self, task: TaskType) -> list[str]:
        """Gets the class names for a particular task.

        @type task: TaskType
        @param task: Task to get the class names for.
        @rtype: list[str]
        @return: Class names for the task.
        """
        return self.dataset_metadata.classes(self.get_task_name(task))

    @property
    def n_keypoints(self) -> int:
        """Getter for the number of keypoints.

        @type: int
        @raises ValueError: If the node does not support keypoints.
        @raises RuntimeError: If the node doesn't define any task.
        """
        if self._n_keypoints is not None:
            return self._n_keypoints

        if self._tasks:
            if TaskType.KEYPOINTS not in self._tasks:
                raise ValueError(f"{self.name} does not support keypoints.")
            return self.dataset_metadata.n_keypoints(
                self.get_task_name(TaskType.KEYPOINTS)
            )

        raise RuntimeError(
            f"{self.name} does not have any tasks defined, "
            "`BaseNode.n_keypoints` property cannot be used. "
            "Either override the `tasks` class attribute, "
            "pass the `n_keypoints` attribute to the constructor or call "
            "the `BaseNode.dataset_metadata.get_n_keypoints` method manually."
        )

    @property
    def n_classes(self) -> int:
        """Getter for the number of classes.

        @type: int
        @raises RuntimeError: If the node doesn't define any task.
        @raises ValueError: If the number of classes is different for
            different tasks. In that case, use the L{get_n_classes}
            method.
        """
        if self._n_classes is not None:
            return self._n_classes

        if not self._tasks:
            raise RuntimeError(
                f"{self.name} does not have any tasks defined, "
                "`BaseNode.n_classes` property cannot be used. "
                "Either override the `tasks` class attribute, "
                "pass the `n_classes` attribute to the constructor or call "
                "the `BaseNode.dataset_metadata.n_classes` method manually."
            )
        elif len(self._tasks) == 1:
            return self.dataset_metadata.n_classes(self.task)
        else:
            n_classes = [
                self.dataset_metadata.n_classes(self.get_task_name(task))
                for task in self._tasks
            ]
            if len(set(n_classes)) == 1:
                return n_classes[0]
            raise ValueError(
                "Node defines multiple tasks but they have different number of classes. "
                "This is likely an error, as the number of classes should be the same."
                "If it is intended, use `BaseNode.get_n_classes` instead."
            )

    @property
    def class_names(self) -> list[str]:
        """Getter for the class names.

        @type: list[str]
        @raises RuntimeError: If the node doesn't define any task.
        @raises ValueError: If the class names are different for
            different tasks. In that case, use the L{get_class_names}
            method.
        """
        if not self._tasks:
            raise RuntimeError(
                f"{self.name} does not have any tasks defined, "
                "`BaseNode.class_names` property cannot be used. "
                "Either override the `tasks` class attribute, "
                "pass the `n_classes` attribute to the constructor or call "
                "the `BaseNode.dataset_metadata.class_names` method manually."
            )
        elif len(self._tasks) == 1:
            return self.dataset_metadata.classes(self.task)
        else:
            class_names = [
                self.dataset_metadata.classes(self.get_task_name(task))
                for task in self._tasks
            ]
            if all(set(names) == set(class_names[0]) for names in class_names):
                return class_names[0]
            raise ValueError(
                "Node defines multiple tasks but they have different class names. "
                "This is likely an error, as the class names should be the same. "
                "If it is intended, use `BaseNode.get_class_names` instead."
            )

    @property
    def input_shapes(self) -> list[Packet[Size]]:
        """Getter for the input shapes.

        @type: list[Packet[Size]]
        @raises RuntimeError: If the C{input_shapes} were not set during
            initialization.
        """

        if self._input_shapes is None:
            raise self._non_set_error("input_shapes")
        return self._input_shapes

    @property
    def original_in_shape(self) -> Size:
        """Getter for the original input shape as [N, H, W].

        @type: Size
        @raises RuntimeError: If the C{original_in_shape} were not set
            during initialization.
        """
        if self._original_in_shape is None:
            raise self._non_set_error("original_in_shape")
        return self._original_in_shape

    @property
    def dataset_metadata(self) -> DatasetMetadata:
        """Getter for the dataset metadata.

        @type: L{DatasetMetadata}
        @raises RuntimeError: If the C{dataset_metadata} were not set
            during initialization.
        """
        if self._dataset_metadata is None:
            raise RuntimeError(
                f"{self._non_set_error('dataset_metadata')}"
                "Either provide `dataset_metadata` or `n_classes`."
            )
        return self._dataset_metadata

    @property
    def in_sizes(self) -> Size | list[Size]:
        """Simplified getter for the input shapes.

        Should work out of the box for most cases where the C{input_shapes} are
        sufficiently simple. Otherwise, the C{input_shapes} should be used directly.

        In case C{in_sizes} were provided during initialization, they are returned
        directly.

        Example::

            >>> input_shapes = [{"features": [Size(64, 128, 128), Size(3, 224, 224)]}]
            >>> attach_index = -1
            >>> in_sizes = Size(3, 224, 224)

            >>> input_shapes = [{"features": [Size(64, 128, 128), Size(3, 224, 224)]}]
            >>> attach_index = "all"
            >>> in_sizes = [Size(64, 128, 128), Size(3, 224, 224)]

        @type: Size | list[Size]
        @raises RuntimeError: If the C{input_shapes} are too complicated for
            the default implementation.
        """
        if self._in_sizes is not None:
            return self._in_sizes

        features = self.input_shapes[0].get("features")
        if features is None:
            raise RuntimeError(
                f"Feature field is missing in {self.name}. "
                "The default implementation of `in_sizes` cannot be used."
            )
        return self.get_attached(self.input_shapes[0]["features"])

    @property
    def in_channels(self) -> int | list[int]:
        """Simplified getter for the number of input channels.

        Should work out of the box for most cases where the
        C{input_shapes} are sufficiently simple. Otherwise, the
        C{input_shapes} should be used directly. If C{attach_index} is
        set to "all" or is a slice, returns a list of input channels,
        otherwise returns a single value.

        @type: int | list[int]
        @raises RuntimeError: If the C{input_shapes} are too complicated
            for the default implementation of C{in_sizes}.
        """
        return self._get_nth_size(-3)

    @property
    def in_height(self) -> int | list[int]:
        """Simplified getter for the input height.

        Should work out of the box for most cases where the
        C{input_shapes} are sufficiently simple. Otherwise, the
        C{input_shapes} should be used directly.

        @type: int | list[int]
        @raises RuntimeError: If the C{input_shapes} are too complicated
            for the default implementation of C{in_sizes}.
        """
        return self._get_nth_size(-2)

    @property
    def in_width(self) -> int | list[int]:
        """Simplified getter for the input width.

        Should work out of the box for most cases where the
        C{input_shapes} are sufficiently simple. Otherwise, the
        C{input_shapes} should be used directly.

        @type: int | list[int]
        @raises RuntimeError: If the C{input_shapes} are too complicated
            for the default implementation of C{in_sizes}.
        """
        return self._get_nth_size(-1)

    def load_checkpoint(self, path: str, strict: bool = True):
        """Loads checkpoint for the module. If path is url then it
        downloads it locally and stores it in cache.

        @type path: str | None
        @param path: Path to local or remote .ckpt file.
        @type strict: bool
        @param strict: Whether to load weights strictly or not. Defaults
            to True.
        """
        local_path = safe_download(url=path)
        if local_path:
            state_dict = torch.load(
                local_path, weights_only=False, map_location="cpu"
            )[
                "state_dict"
            ]  # load explicitly to cpu, PL takes care of transfering to CUDA is needed
            self.load_state_dict(state_dict, strict=strict)
            logging.info(f"Checkpoint for {self.name} loaded.")
        else:
            logger.warning(
                f"No checkpoint available for {self.name}, skipping."
            )

    @property
    def export(self) -> bool:
        """Getter for the export mode."""
        return self._export

    def set_export_mode(self, mode: bool = True) -> None:
        """Sets the module to export mode.

        @type mode: bool
        @param mode: Value to set the export mode to. Defaults to True.
        """
        self._export = mode

    @property
    def remove_on_export(self) -> bool:
        """Getter for the remove_on_export attribute."""
        return self._remove_on_export

    @property
    def export_output_names(self) -> list[str] | None:
        """Getter for the export_output_names attribute."""
        return self._export_output_names

    def unwrap(self, inputs: list[Packet[Tensor]]) -> ForwardInputT:
        """Prepares inputs for the forward pass.

        Unwraps the inputs from the C{list[Packet[Tensor]]} input so
        they can be passed to the forward call. The default
        implementation expects a single input with C{features} key and
        returns the tensor or tensors at the C{attach_index} position.

        For most cases the default implementation should be sufficient.
        Exceptions are modules with multiple inputs or producing more
        complex outputs. This is typically the case for output nodes.

        @type inputs: list[Packet[Tensor]]
        @param inputs: Inputs to the node.
        @rtype: ForwardInputT
        @return: Prepared inputs, ready to be passed to the L{forward}
            method.
        @raises ValueError: If the number of inputs is not equal to 1.
            In such cases the method has to be overridden.
        """
        if len(inputs) > 1:
            raise ValueError(
                f"Node {self.name} expects a single input, but got {len(inputs)} inputs instead. "
                "If the node expects multiple inputs, the `unwrap` method should be overridden."
            )
        return self.get_attached(inputs[0]["features"])  # type: ignore

    @abstractmethod
    def forward(self, inputs: ForwardInputT) -> ForwardOutputT:
        """Forward pass of the module.

        @type inputs: L{ForwardInputT}
        @param inputs: Inputs to the module.
        @rtype: L{ForwardOutputT}
        @return: Result of the forward pass.
        """
        ...

    def wrap(self, output: ForwardOutputT) -> Packet[Tensor]:
        """Wraps the output of the forward pass into a
        C{Packet[Tensor]}.

        The default implementation expects a single tensor or a list of tensors
        and wraps them into a Packet with either the node task as a key
        or "features" key if task is not defined.

        Example::

            >>> class FooNode(BaseNode):
            ...     tasks = [TaskType.CLASSIFICATION]
            ...
            ... class BarNode(BaseNode):
            ...     pass
            ...
            >>> node = FooNode()
            >>> node.wrap(torch.rand(1, 10))
            {"classification": [Tensor(1, 10)]}
            >>> node = BarNode()
            >>> node.wrap([torch.rand(1, 10), torch.rand(1, 10)])
            {"features": [Tensor(1, 10), Tensor(1, 10)]}

        @type output: ForwardOutputT
        @param output: Output of the forward pass.

        @rtype: L{Packet}[Tensor]
        @return: Wrapped output.

        @raises ValueError: If the C{output} argument is not a tensor or a list of tensors.
            In such cases the L{wrap} method should be overridden.
        """

        if isinstance(output, Tensor):
            outputs = [output]
        elif isinstance(output, (list, tuple)) and all(
            isinstance(t, Tensor) for t in output
        ):
            outputs = list(output)
        else:
            raise ValueError(
                "Default `wrap` expects a single tensor or a list of tensors."
            )
        try:
            task = self.task
        except RuntimeError:
            task = "features"
        return {task: outputs}

    def run(self, inputs: list[Packet[Tensor]]) -> Packet[Tensor]:
        """Combines the forward pass with the wrapping and unwrapping of
        the inputs.

        @type inputs: list[Packet[Tensor]]
        @param inputs: Inputs to the module.

        @rtype: L{Packet}[Tensor]
        @return: Outputs of the module as a dictionary of list of tensors:
            C{{"features": [Tensor, ...], "segmentation": [Tensor]}}

        @raises RuntimeError: If default L{wrap} or L{unwrap} methods are not sufficient.
        """
        unwrapped = self.unwrap(inputs)
        outputs = self(unwrapped)
        wrapped = self.wrap(outputs)
        str_tasks = [task.value for task in self._tasks] if self._tasks else []
        for key in list(wrapped.keys()):
            if key in str_tasks:
                value = wrapped.pop(key)
                wrapped[self.get_task_name(TaskType(key))] = value
        return wrapped

    T = TypeVar("T", Tensor, Size)

    def get_attached(self, lst: list[T]) -> list[T] | T:
        """Gets the attached elements from a list.

        This method is used to get the attached elements from a list
        based on the C{attach_index} attribute.

        @type lst: list[T]
        @param lst: List to get the attached elements from. Can be
            either a list of tensors or a list of sizes.
        @rtype: list[T] | T
        @return: Attached elements. If C{attach_index} is set to
            C{"all"} or is a slice, returns a list of attached elements.
        @raises ValueError: If the C{attach_index} is invalid.
        """

        def _normalize_index(index: int) -> int:
            if index < 0:
                index += len(lst)
            return index

        def _normalize_slice(i: int, j: int) -> slice:
            if i < 0 and j < 0:
                return slice(len(lst) + i, len(lst) + j, -1 if i > j else 1)
            if i < 0:
                return slice(len(lst) + i, j, 1)
            if j < 0:
                return slice(i, len(lst) + j, 1)
            if i > j:
                return slice(i, j, -1)
            return slice(i, j, 1)

        match self.attach_index:
            case "all":
                return lst
            case int(i):
                i = _normalize_index(i)
                if i >= len(lst):
                    raise ValueError(
                        f"Attach index {i} is out of range for list of length {len(lst)}."
                    )
                return lst[_normalize_index(i)]
            case (int(i), int(j)):
                return lst[_normalize_slice(i, j)]
            case (int(i), int(j), int(k)):
                return lst[i:j:k]
            case _:
                raise ValueError(
                    f"Invalid attach index: `{self.attach_index}`"
                )

    def _get_nth_size(self, idx: int) -> int | list[int]:
        match self.in_sizes:
            case Size(sizes):
                return sizes[idx]
            case list(sizes):
                return [size[idx] for size in sizes]

    def _non_set_error(self, name: str) -> RuntimeError:
        return RuntimeError(
            f"'{self.name}' node is trying to access `{name}`, "
            "but it was not set during initialization. "
        )
