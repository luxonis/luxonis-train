import inspect
import logging
from abc import ABC, abstractmethod
from contextlib import suppress
from operator import itemgetter
from typing import Generic, TypeVar

import torch
from bidict import bidict
from loguru import logger
from luxonis_ml.utils.registry import AutoRegisterMeta
from torch import Size, Tensor, nn
from typeguard import TypeCheckError, check_type, typechecked

from luxonis_train.registry import NODES
from luxonis_train.tasks import Task
from luxonis_train.typing import AttachIndexType, Packet
from luxonis_train.utils import (
    DatasetMetadata,
    IncompatibleError,
    safe_download,
)

ForwardOutputT = TypeVar("ForwardOutputT")
ForwardInputT = TypeVar("ForwardInputT")


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
        - L{task}: An instance of `luxonis_train.tasks.Task` that specifies the
            task of the node. Usually defined for head nodes.

    Example::
        class MyNode(BaseNode):
            task = Tasks.CLASSIFICATION

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
    """

    attach_index: AttachIndexType = None
    task: Task | None = None

    @typechecked
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
        task_name: str | None = None,
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
        @type task_name: str | None
        @param task_name: Specifies which task group from the dataset to use
            in case the dataset contains multiple tasks. Otherwise, the
            task group is inferred from the dataset metadata.
        """
        super().__init__()

        if attach_index is not None:
            logger.warning(
                f"Node {self.name} overrides `attach_index` "
                f"by setting it to '{attach_index}'. "
                "Make sure this is intended."
            )
            self.attach_index = attach_index

        if self.attach_index is None:
            parameters = inspect.signature(self.forward).parameters
            assert parameters, f"`{self.name}.forward` has no parameters."

            annotation = next(iter(parameters.values())).annotation

            if len(parameters) > 1 or annotation is inspect.Parameter.empty:
                logger.warning(self._missing_attach_index_message())
            elif annotation == Tensor:
                self.attach_index = -1
            elif annotation == list[Tensor]:
                self.attach_index = "all"
            else:
                logger.warning(self._missing_attach_index_message())

        if task_name is None and dataset_metadata is not None:
            if len(dataset_metadata.task_names) == 1:
                task_name = next(iter(dataset_metadata.task_names))
            else:
                raise ValueError(
                    f"Dataset contain multiple tasks, but the `task_name` "
                    f"argument for node '{self.name}' was not provided."
                )
        self.task_name = task_name or ""

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

        self.current_epoch = 0

        self._check_type_overrides()

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def n_keypoints(self) -> int:
        """Getter for the number of keypoints.

        @type: int
        @raises ValueError: If the node does not support keypoints.
        """
        if self._n_keypoints is not None:
            return self._n_keypoints

        return self.dataset_metadata.n_keypoints(self.task_name)

    @property
    def n_classes(self) -> int:
        """Getter for the number of classes.

        @type: int
        """
        if self._n_classes is not None:
            return self._n_classes

        return self.dataset_metadata.n_classes(self.task_name)

    @property
    def classes(self) -> bidict[str, int]:
        """Getter for the class mappings.

        @type: dict[str, int]
        """
        return self.dataset_metadata.classes(self.task_name)

    @property
    def class_names(self) -> list[str]:
        """Getter for the class names.

        @type: list[str]
        """
        return [
            name for name, _ in sorted(self.classes.items(), key=itemgetter(1))
        ]

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
            raise RuntimeError(self._non_set_error("dataset_metadata"))
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
        assert isinstance(features, list)
        return self.get_attached(features)

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

    def load_checkpoint(
        self, path: str | None = None, strict: bool = True
    ) -> None:
        """Loads checkpoint for the module.

        @type path: str | None
        @param path: Path to local or remote .ckpt file.
        @type strict: bool
        @param strict: Whether to load weights strictly or not. Defaults
            to True.
        """
        path = path or self.get_weights_url()
        if path is None:
            raise ValueError(
                f"Attempting to load weights for '{self.name}' "
                f"node, but the `path` argument was not provided and "
                "the node does not implement the `get_weights_url` method."
            )

        local_path = safe_download(url=path)
        if local_path:
            # load explicitly to cpu, PL takes care of transfering to CUDA is needed
            state_dict = torch.load(  # nosemgrep
                local_path, weights_only=False, map_location="cpu"
            )["state_dict"]
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
        Exceptions are modules with multiple inputs.

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
                f"Node {self.name} expects a single input, "
                f"but got {len(inputs)} inputs instead. "
                "If the node expects multiple inputs, "
                "the `unwrap` method should be overridden."
            )
        inp = inputs[0]["features"]
        if isinstance(inp, Tensor):
            return inp  # type: ignore
        return self.get_attached(inp)  # type: ignore

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
            ...     task = Tasks.CLASSIFICATION
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
            outputs = output
        elif isinstance(output, list | tuple) and all(
            isinstance(t, Tensor) for t in output
        ):
            outputs = list(output)
        else:
            raise ValueError(
                "Default `wrap` expects a single tensor or a list of tensors."
            )
        if self.task is None:
            name = "features"
        else:
            name = self.task.main_output
        return {name: outputs}

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
        return self.wrap(self(self.unwrap(inputs)))

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

        def _normalize_slice(i: int, j: int, k: int | None = None) -> slice:
            if i < 0 and j < 0:
                return slice(
                    len(lst) + i, len(lst) + j, k or -1 if i > j else 1
                )
            if i < 0:
                return slice(len(lst) + i, j, k or 1)
            if j < 0:
                return slice(i, len(lst) + j, k or 1)
            if i > j:
                return slice(i, j, k or -1)
            return slice(i, j, k or 1)

        match self.attach_index:
            case "all":
                return lst
            case int(i):
                i = _normalize_index(i)
                if i >= len(lst):
                    raise ValueError(
                        f"Attach index {i} is out of range "
                        f"for list of length {len(lst)}."
                    )
                return lst[i]
            case (int(i), int(j)):
                return lst[_normalize_slice(i, j)]
            case (int(i), int(j), int(k)):
                return lst[_normalize_slice(i, j, k)]
            case None:
                raise RuntimeError(self._missing_attach_index_message())

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

    def _missing_attach_index_message(self) -> str:
        return (
            f"Attach index not defined for node '{self.name}'  "
            "and could not be inferred. "
            "Some parts of the framework will not work. "
            "Either pass `attach_index` to the base constructor, "
            "define it as a class atrribute, or provide proper "
            "type hints for the `forward` method for implicit inference"
        )

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
                        raise IncompatibleError(
                            f"Node '{self.name}' specifies the type of "
                            f"the property `{name}` as `{typ}`, "
                            f"but received `{type(value)}`. "
                            f"This may indicate that the '{self.name}' node is "
                            "not compatible with its predecessor."
                        ) from e
