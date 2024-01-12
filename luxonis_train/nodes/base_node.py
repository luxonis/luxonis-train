from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from luxonis_ml.utils.registry import AutoRegisterMeta
from pydantic import BaseModel, ValidationError
from torch import Size, Tensor, nn

from luxonis_train.utils.general import DatasetMetadata, validate_packet
from luxonis_train.utils.registry import NODES
from luxonis_train.utils.types import (
    AttachIndexType,
    FeaturesProtocol,
    IncompatibleException,
    LabelType,
    Packet,
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

    Each node can define a list of L{BaseProtocol}s that the inputs must conform to.
    L{BaseProtocol} is a pydantic model that defines the structure of the input.
    When the node is called, the inputs are validated against the protocols and
    then sent to the L{unwrap} method. The C{unwrap} method should return a valid
    input to the L{forward} method. Outputs of the C{forward} method are then
    send to L{weap} method, which wraps the output into a C{Packet}, which is the
    output of the node.

    The L{run} method combines the C{unwrap}, C{forward} and C{wrap} methods
    together with input validation.


    @type input_shapes: list[Packet[Size]] | None
    @param input_shapes: List of input shapes for the module.

    @type original_in_shape: Size | None
    @param original_in_shape: Original input shape of the model. Some
        nodes won't function if not provided.

    @type dataset_metadata: L{DatasetMetadata} | None
    @param dataset_metadata: Metadata of the dataset.
        Some nodes won't function if not provided.

    @type attach_index: AttachIndexType
    @param attach_index: Index of previous output that this node attaches to.
        Can be a single integer to specify a single output, a tuple of
        two or three integers to specify a range of outputs or `"all"` to
        specify all outputs. Defaults to "all". Python indexing conventions apply.

    @type in_protocols: list[type[BaseModel]]
    @param in_protocols: List of input protocols used to validate inputs to the node.
        Defaults to [FeaturesProtocol].

    @type n_classes: int | None
    @param n_classes: Number of classes in the dataset. Provide only
        in case `dataset_metadata` is not provided. Defaults to None.

    @type in_sizes: Size | list[Size] | None
    @param in_sizes: List of input sizes for the node.
        Provide only in case the `input_shapes` were not provided.
    """

    attach_index: AttachIndexType = "all"

    def __init__(
        self,
        *,
        input_shapes: list[Packet[Size]] | None = None,
        original_in_shape: Size | None = None,
        dataset_metadata: DatasetMetadata | None = None,
        attach_index: AttachIndexType | None = None,
        in_protocols: list[type[BaseModel]] | None = None,
        n_classes: int | None = None,
        in_sizes: Size | list[Size] | None = None,
        task_type: LabelType | None = None,
    ):
        super().__init__()

        self.attach_index = attach_index or self.attach_index
        self.in_protocols = in_protocols or [FeaturesProtocol]
        self.task_type = task_type

        self._input_shapes = input_shapes
        self._original_in_shape = original_in_shape
        if n_classes is not None:
            if dataset_metadata is not None:
                raise ValueError("Cannot set both `dataset_metadata` and `n_classes`.")
            dataset_metadata = DatasetMetadata(n_classes=n_classes)
        self._dataset_metadata = dataset_metadata
        self._export = False
        self._epoch = 0
        self._in_sizes = in_sizes

    def _non_set_error(self, name: str) -> ValueError:
        return ValueError(
            f"{self.__class__.__name__} is trying to access `{name}`, "
            "but it was not set during initialization. "
        )

    @property
    def n_classes(self) -> int:
        """Getter for the number of classes."""
        return self.dataset_metadata.n_classes(self.task_type)

    @property
    def class_names(self) -> list[str]:
        """Getter for the class names."""
        return self.dataset_metadata.class_names(self.task_type)

    @property
    def input_shapes(self) -> list[Packet[Size]]:
        """Getter for the input shapes."""
        if self._input_shapes is None:
            raise self._non_set_error("input_shapes")
        return self._input_shapes

    @property
    def original_in_shape(self) -> Size:
        """Getter for the original input shape."""
        if self._original_in_shape is None:
            raise self._non_set_error("original_in_shape")
        return self._original_in_shape

    @property
    def dataset_metadata(self) -> DatasetMetadata:
        """Getter for the dataset metadata.

        @type: L{DatasetMetadata}
        @raises ValueError: If the C{dataset_metadata} is C{None}.
        """
        if self._dataset_metadata is None:
            raise ValueError(
                f"{self._non_set_error('dataset_metadata')}"
                "Either provide `dataset_metadata` or `n_classes`."
            )
        return self._dataset_metadata

    @property
    def in_sizes(self) -> Size | list[Size]:
        """Simplified getter for the input shapes.

        Should work out of the box for most cases where the `input_shapes` are
        sufficiently simple. Otherwise the `input_shapes` should be used directly.

        In case `in_sizes` were provided during initialization, they are returned
        directly.

        Example:

            >>> input_shapes = [{"features": [Size(1, 64, 128, 128), Size(1, 3, 224, 224)]}]
            >>> attach_index = -1
            >>> in_sizes = Size(1, 3, 224, 224)

            >>> input_shapes = [{"features": [Size(1, 64, 128, 128), Size(1, 3, 224, 224)]}]
            >>> attach_index = "all"
            >>> in_sizes = [Size(1, 64, 128, 128), Size(1, 3, 224, 224)]

        @type: Size | list[Size]
        @raises IncompatibleException: If the C{input_shapes} are too complicated for
            the default implementation.
        """
        if self._in_sizes is not None:
            return self._in_sizes

        features = self.input_shapes[0].get("features")
        if features is None:
            raise IncompatibleException(
                f"Feature field is missing in {self.__class__.__name__}. "
                "The default implementation of `in_sizes` cannot be used."
            )
        shapes = self.get_attached(self.input_shapes[0]["features"])
        if isinstance(shapes, list) and len(shapes) == 1:
            return shapes[0]
        return shapes

    @property
    def in_channels(self) -> int | list[int]:
        """Simplified getter for the number of input channels.

        Should work out of the box for most cases where the C{input_shapes} are
        sufficiently simple. Otherwise the C{input_shapes} should be used directly. If
        C{attach_index} is set to "all" or is a slice, returns a list of input channels,
        otherwise returns a single value.

        @type: int | list[int]
        @raises IncompatibleException: If the C{input_shapes} are too complicated for
            the default implementation.
        """
        return self._get_nth_size(1)

    @property
    def in_height(self) -> int | list[int]:
        """Simplified getter for the input height.

        Should work out of the box for most cases where the `input_shapes` are
        sufficiently simple. Otherwise the `input_shapes` should be used directly.

        @type: int | list[int]
        @raises IncompatibleException: If the C{input_shapes} are too complicated for
            the default implementation.
        """
        return self._get_nth_size(2)

    @property
    def in_width(self) -> int | list[int]:
        """Simplified getter for the input width.

        Should work out of the box for most cases where the `input_shapes` are
        sufficiently simple. Otherwise the `input_shapes` should be used directly.

        @type: int | list[int]
        @raises IncompatibleException: If the C{input_shapes} are too complicated for
            the default implementation.
        """
        return self._get_nth_size(3)

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

    def unwrap(self, inputs: list[Packet[Tensor]]) -> ForwardInputT:
        """Prepares inputs for the forward pass.

        Unwraps the inputs from the C{list[Packet[Tensor]]} input so they can be passed
        to the forward call. The default implementation expects a single input with
        C{features} key and returns the tensor or tensors at the C{attach_index}
        position.

        For most cases the default implementation should be sufficient. Exceptions are
        modules with multiple inputs or producing more complex outputs. This is
        typically the case for output nodes.

        @type inputs: list[Packet[Tensor]]
        @param inputs: Inputs to the node.
        @rtype: ForwardInputT
        @return: Prepared inputs, ready to be passed to the L{forward} method.
        """
        return self.get_attached(inputs[0]["features"])  # type: ignore

    @abstractmethod
    def forward(self, inputs: ForwardInputT) -> ForwardOutputT:
        """Forward pass of the module.

        @type inputs: ForwardInputT
        @param inputs: Inputs to the module.
        @rtype: ForwardOutputT
        @return: Result of the forward pass.
        """
        ...

    def wrap(self, output: ForwardOutputT) -> Packet[Tensor]:
        """Wraps the output of the forward pass into a `Packet[Tensor]`.

        The default implementation expects a single tensor or a list of tensors
        and wraps them into a Packet with `features` key.

        @type output: ForwardOutputT
        @param output: Output of the forward pass.

        @rtype: L{Packet}[Tensor]
        @return: Wrapped output.
        """

        match output:
            case Tensor(data=out):
                outputs = [out]
            case list(tensors) if all(isinstance(t, Tensor) for t in tensors):
                outputs = tensors
            case _:
                raise IncompatibleException(
                    "Default `wrap` expects a single tensor or a list of tensors."
                )
        return {"features": outputs}

    def run(self, inputs: list[Packet[Tensor]]) -> Packet[Tensor]:
        """Combines the forward pass with the wrapping and unwrapping of the inputs.

        Additionally validates the inputs against `in_protocols`.

        @type inputs: list[Packet[Tensor]]
        @param inputs: Inputs to the module.

        @rtype: L{Packet}[Tensor]
        @return: Outputs of the module as a dictionary of list of tensors:
            `{"features": [Tensor, ...], "segmentation": [Tensor]}`

        @raises IncompatibleException: If the inputs are not compatible with the node.
        """
        unwrapped = self.unwrap(self.validate(inputs))
        outputs = self(unwrapped)
        return self.wrap(outputs)

    def validate(self, data: list[Packet[Tensor]]) -> list[Packet[Tensor]]:
        """Validates the inputs against `in_protocols`."""
        if len(data) != len(self.in_protocols):
            raise IncompatibleException(
                f"Node {self.__class__.__name__} expects {len(self.in_protocols)} inputs, "
                f"but got {len(data)} inputs instead."
            )
        try:
            return [
                validate_packet(d, protocol)
                for d, protocol in zip(data, self.in_protocols)
            ]
        except ValidationError as e:
            raise IncompatibleException.from_validation_error(
                e, self.__class__.__name__
            ) from e

    T = TypeVar("T", Tensor, Size)

    def get_attached(self, lst: list[T]) -> list[T] | T:
        """Gets the attached elements from a list.

        This method is used to get the attached elements from a list based on
        the `attach_index` attribute.

        @type lst: list[T]
        @param lst: List to get the attached elements from. Can be either
            a list of tensors or a list of sizes.

        @rtype: list[T] | T
        @return: Attached elements. If `attach_index` is set to `"all"` or is a slice,
            returns a list of attached elements.

        @raises ValueError: If the `attach_index` is invalid.
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
                raise ValueError(f"Invalid attach index: `{self.attach_index}`")

    def _get_nth_size(self, idx: int) -> int | list[int]:
        match self.in_sizes:
            case Size(sizes):
                return sizes[idx]
            case list(sizes):
                return [size[idx] for size in sizes]
