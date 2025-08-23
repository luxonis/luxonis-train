import inspect
import logging
from abc import ABC, abstractmethod
from contextlib import suppress
from operator import itemgetter
from typing import Literal, TypeVar

import torch
from bidict import bidict
from loguru import logger
from luxonis_ml.typing import Kwargs, check_type
from torch import Size, Tensor, nn
from typeguard import typechecked

from luxonis_train.config.predefined_models.base_predefined_model import (
    VariantMeta,
)
from luxonis_train.nodes.blocks.reparametrizable import Reparametrizable
from luxonis_train.registry import NODES
from luxonis_train.tasks import Task
from luxonis_train.typing import AttachIndexType, Packet, get_signature
from luxonis_train.utils import (
    DatasetMetadata,
    IncompatibleError,
    safe_download,
)


class PostInitMeta(VariantMeta):
    def __call__(cls, *args, **kwargs):
        obj = VariantMeta.__call__(cls, *args, **kwargs)
        if isinstance(obj, cls):
            post_init = getattr(obj, "_post_init", None)
            if callable(post_init):
                post_init()
        return obj


class BaseNode(
    nn.Module,
    ABC,
    metaclass=PostInitMeta,
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

    When subclassing, the following methods should be implemented:
        - L{forward}: Forward pass of the module.

    Additionally, the following class attributes can be defined:
        - L{attach_index}: Index of previous output that this node attaches to.
        - L{task}: An instance of `luxonis_train.tasks.Task` that specifies the
            task of the node. Usually defined for head nodes.

    @type attach_index: AttachIndexType
    @ivar attach_index: Index of previous output that this node attaches to.
        Can be a single integer to specify a single output, a tuple of
        two or three integers to specify a range of outputs or C{"all"} to
        specify all outputs. Defaults to "all". Python indexing conventions apply.
    """

    attach_index: AttachIndexType = None
    task: Task | None = None

    _variant: str | None

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
        weights: str | Literal["download", "yolo", "default"] = "default",
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
        self._in_sizes = in_sizes
        self._weights = weights

        self.current_epoch = 0

        self._check_type_overrides()

    def _post_init(self) -> None:
        if self._weights == "default":
            return

        if self._weights == "download":
            self.load_checkpoint()
        elif self._weights.startswith("http"):
            self.load_checkpoint(ckpt=self._weights)
        else:
            self.initialize_weights(method=self._weights)

    def initialize_weights(
        self, method: Literal["yolo"] | str | None = None
    ) -> None:
        """Initializes the weights of the module.

        This method should be overridden in subclasses to provide custom
        weight initialization.

        @type method: str | None
        @param method: Method to use for weight initialization. If set
            to "yolo", the weights are initialized using the YOLOv5
            method. Defaults to None, which does not perform any
            initialization.
        """
        if method == "yolo":
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 3e-2
                elif isinstance(
                    m,
                    nn.Hardswish | nn.LeakyReLU | nn.ReLU | nn.ReLU6 | nn.SiLU,
                ):
                    m.inplace = True

    @staticmethod
    def get_variants() -> tuple[str, dict[str, Kwargs]]:
        """Returns a name of the default varaint and a dictionary of
        available model variants with their parameters.

        The keys are the variant names, and the values are dictionaries
        of parameters which can be used as C{**kwargs} for the
        predefined model constructor.

        @rtype: tuple[str, dict[str, Params]]
        @return: A tuple containing the default variant name and a
            dictionary of available variants with their parameters.
        """
        raise NotImplementedError

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def variant(self) -> str:
        if self._variant is None:
            raise RuntimeError(f"Variant was not set for node '{self.name}'.")
        return self._variant

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

    def get_weights_url(self) -> str | None:
        """Returns the URL to the weights of the node.

        Subclasses can override this method to provide a URL to support
        loading weights from a remote location.

        It is possible to use a special placeholder C{{github}} in the
        URL, which will be replaced with
        C{"https://github.com/luxonis/luxonis-train/releases/download/{version}"},
        where C{{version}} is the version of `luxonis-train` library.

        The file pointed to by the URL should be a C{.ckpt} file
        that is directly loadable using C{nn.Module.load_state_dict}.
        """
        return None

    def _get_weights_url(self) -> str | None:
        url = self.get_weights_url()
        if url is None:
            return None

        return url.replace(
            "{github}",
            "gcs://luxonis-test-data/weights/v0.4.0-beta/"
            "releases/download/v0.2.1-beta",
        )

    def load_checkpoint(
        self,
        ckpt: str | dict[str, Tensor] | None = None,
        *,
        strict: bool = True,
    ) -> None:
        """Loads checkpoint for the module.

        @type ckpt: str | dict[str, Tensor] | None
        @param ckpt: Path to local or remote .ckpt file.
        @type strict: bool
        @param strict: Whether to load weights strictly or not. Defaults
            to True.
        """
        ckpt = ckpt or self._get_weights_url()
        if not isinstance(ckpt, dict):
            logger.info(f"Loading weights from '{ckpt}'")
        if ckpt is None:
            raise ValueError(
                f"Attempting to load weights for '{self.name}' "
                f"node, but the `ckpt` argument was not provided and "
                "the node does not implement the `get_weights_url` method."
            )

        if isinstance(ckpt, dict):
            state_dict = ckpt
        else:
            local_path = safe_download(url=ckpt)
            if local_path:
                # load explicitly to cpu, PL takes care of transfering to CUDA is needed
                state_dict = torch.load(  # nosemgrep
                    local_path, weights_only=False, map_location="cpu"
                )["state_dict"]
            else:
                logger.warning(
                    f"No checkpoint available for {self.name}, skipping."
                )
                return

        self.load_state_dict(state_dict, strict=strict)
        logging.info(f"Checkpoint for {self.name} loaded.")

    @property
    def export(self) -> bool:
        """Getter for the export mode."""
        return self._export

    @export.setter
    def export(self, mode: bool) -> None:
        """Sets the module to export mode."""
        self.set_export_mode(mode)

    def set_export_mode(self, /, mode: bool) -> None:
        """Sets the module to export mode.

        @type mode: bool
        @param mode: Value to set the export mode to.
        """
        self._export = mode
        if mode:
            logger.info(f"Reparametrizing '{self.name}'")
        else:
            logger.info(f"Restoring reparametrized '{self.name}'")

        for name, module in self.named_modules():
            if isinstance(module, Reparametrizable):
                if mode:
                    logger.debug(f"Reparametrizing '{name}' in '{self.name}'")
                    module.reparametrize()
                else:
                    logger.debug(
                        f"Restoring reparametrized '{name}' in '{self.name}'"
                    )
                    module.restore()

    @property
    def remove_on_export(self) -> bool:
        """Getter for the remove_on_export attribute."""
        return self._remove_on_export

    @property
    def export_output_names(self) -> list[str] | None:
        """Getter for the export_output_names attribute."""
        return self._export_output_names

    @abstractmethod
    def forward(
        self, inputs: Tensor | list[Tensor]
    ) -> Tensor | list[Tensor] | Packet[Tensor]:
        """Forward pass of the module.

        @type inputs: L{ForwardInputT}
        @param inputs: Inputs to the module.
        @rtype: Tensor | list[Tensor] | Packet[Tensor]
        @return: Result of the forward pass.
        """
        ...

    def run(self, inputs: list[Packet[Tensor]]) -> Packet[Tensor]:
        """Combines the forward pass with the wrapping and unwrapping of
        the inputs.

        @type inputs: list[Packet[Tensor]]
        @param inputs: Inputs to the module.

        @rtype: L{Packet}[Tensor]
        @return: Outputs of the module as a dictionary of list of tensors:
            C{{"features": [Tensor, ...], "segmentation": [Tensor]}}
        """
        kwargs = {}
        signature = get_signature(self.forward)
        for name, param in signature.items():
            if name in {"input", "inputs", "x"}:
                if len(inputs) != 1:
                    raise RuntimeError(
                        f"Node '{self.name}' expects a single input, "
                        "but is connected to multiple preceding nodes. "
                    )
                input_name = "features"

                packet = inputs[0]
                if input_name not in packet:
                    raise RuntimeError(
                        f"Node '{self.name}' expects an input with key "
                        f"'{input_name}', but it was not found in the packet."
                    )
                value = packet[input_name]
                if isinstance(value, Tensor):
                    if param.annotation != Tensor:
                        raise RuntimeError(
                            f"Node '{self.name}' expects an input with key "
                            f"'{input_name}' to be of type `{param.annotation}`, "
                            "but got a single tensor instead."
                        )
                    kwargs[name] = value
                else:
                    kwargs[name] = self.get_attached(value)

        outputs = self(**kwargs)

        if check_type(outputs, Packet[Tensor]):
            return outputs

        name = "features" if self.task is None else self.task.main_output

        if isinstance(outputs, Tensor):
            return {name: outputs}

        if check_type(outputs, list[Tensor]):
            return {name: outputs}

        raise ValueError(
            "Invalid output type from the forward pass. "
            "Expected Tensor, list of Tensors or a dictionary, "
            f"but got {type(outputs)} instead."
        )

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
                if i < j:
                    return slice(
                        max(len(lst) + i + 1, 0),
                        len(lst) + j + 1,
                        k or -1 if i > j else 1,
                    )
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
                    if not check_type(value, typ):
                        raise IncompatibleError(
                            f"Node '{self.name}' specifies the type of "
                            f"the property `{name}` as `{typ}`, "
                            f"but received `{type(value)}`. "
                            f"This may indicate that the '{self.name}' node is "
                            "not compatible with its predecessor."
                        )
