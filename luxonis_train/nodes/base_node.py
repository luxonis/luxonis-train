"""The base class every node inherits.

`BaseNode` gives a node the shapes of its inputs, so that the node can
size its layers. It maps the input packets to the parameters of
``forward``. It also builds a node from a named variant, loads
pretrained weights, and switches export mode.

"""

import inspect
import logging
import re
from abc import abstractmethod
from contextlib import suppress
from operator import itemgetter
from typing import Literal, TypeVar

import torch
from bidict import bidict
from loguru import logger
from luxonis_ml.typing import Kwargs, check_type
from torch import Size, Tensor, nn
from typeguard import typechecked

from luxonis_train.nodes.blocks.reparameterizable import Reparameterizable
from luxonis_train.registry import NODES
from luxonis_train.tasks import Task
from luxonis_train.typing import AttachIndexType, Packet
from luxonis_train.utils import (
    DatasetMetadata,
    IncompatibleError,
    get_signature,
    safe_download,
)
from luxonis_train.variants import VariantBase

# Value types that `run` can pass to the `forward` method.
_ForwardInput = Tensor | list[Tensor] | Packet[Tensor] | list[Packet[Tensor]]


class BaseNode(nn.Module, VariantBase, register=False, registry=NODES):
    """Base class for all nodes of the model graph.

    A node is a `torch.nn.Module` that reads packets and returns a
    packet. A *packet* is a dictionary that maps an output name to a
    tensor or to a list of tensors. The model calls `run` with one packet
    for each input of the node. The packet of an input node is its
    output. The packet of a loader input holds the tensor in a list under
    the key ``"features"``. `run` passes the packet values to `forward`
    and returns the result as a packet.

    Every subclass registers itself in `luxonis_train.registry.NODES`
    under its class name, so a config refers to the node by that name.
    A ``register_name`` in the class statement replaces the class name.
    Put ``register=False`` in the class statement to skip the
    registration.

    A subclass must implement `forward`. It can also do these steps:

    - Set the class attributes ``attach_index`` and ``task``.
    - Override `get_variants` to declare variants.
    - Override `get_weights_url` to offer pretrained weights.
    - Override `initialize_weights` to initialize its own layers.
    - Annotate a property in the class body, for example
      ``in_channels: int``. The constructor then compares the value of
      the property with the annotation. On a mismatch, it raises
      `IncompatibleError`. The constructor skips the check when the
      property raises ``RuntimeError``. Any other error of the property
      goes out of the constructor. The check reads only the annotations
      of the nearest class that has annotations. Thus the annotations
      of a subclass hide the annotations of its parent.

    Attributes:
        attach_index (AttachIndexType): The output or outputs of the input
            node that the node reads. `get_attached` applies it. The value
            is an integer index, a tuple of two or three integers for a
            range, or ``"all"`` for every output. ``-1`` is the last
            output.

            When a subclass leaves it ``None``, the constructor infers it
            from ``forward``. A ``forward`` with one parameter annotated
            ``Tensor`` gives ``-1``. One parameter annotated
            ``list[Tensor]`` gives ``"all"``. For any other ``forward``
            with parameters, the constructor logs a warning and the index
            stays ``None``.
        task (Task | None): The task of the node. A head sets it. When
            ``forward`` returns a tensor or a list of tensors, `run`
            puts the result under the key ``task.main_output``. When the
            task is ``None``, the key is ``"features"``.
        task_name (str): The dataset task of the node. It is ``""`` when
            the constructor gets no ``task_name``.
        current_epoch (int): The number of the current training epoch,
            from ``0``. `LuxonisLightningModule` sets it at the start of
            each training epoch.

    Example:
        A node that sizes its layer from the input shapes. The
        ``register=False`` keeps the example out of the registry.

        >>> import torch
        >>> from torch import Size, Tensor, nn
        >>> from luxonis_train.nodes import BaseNode
        >>> class Conv(BaseNode, register=False):
        ...     def __init__(self, **kwargs):
        ...         super().__init__(**kwargs)
        ...         self.conv = nn.Conv2d(self.in_channels, 8, kernel_size=1)
        ...
        ...     def forward(self, x: Tensor) -> Tensor:
        ...         return self.conv(x)
        >>> node = Conv(input_shapes=[{"features": [Size([2, 3, 32, 32])]}])
        >>> node.attach_index, node.in_channels
        (-1, 3)
        >>> packet = node.run([{"features": [torch.zeros(2, 3, 32, 32)]}])
        >>> packet["features"].shape
        torch.Size([2, 8, 32, 32])

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
        weights: str | Literal["download", "yolo", "none"] | None = None,
    ):
        """Initialize the node.

        All arguments are keyword-only and optional. A property that
        needs a missing argument raises a ``RuntimeError`` on access.
        For example, `input_shapes` raises it when ``input_shapes`` is
        ``None``. The ``typeguard`` decorator checks the argument types.
        On a mismatch, it raises ``typeguard.TypeCheckError``.

        Args:
            input_shapes (``list[Packet[Size]] | None``): One shape
                packet for each input of the node, in the order of the
                inputs. The shape properties, such as `in_channels`, read
                it.
            original_in_shape (``Size | None``): The shape of the model
                input image, ``[C, H, W]``, without the batch dimension.
            dataset_metadata (DatasetMetadata | None): The metadata of
                the dataset. `n_classes`, `n_keypoints`, `classes`, and
                `class_names` read it.
            n_classes (int | None): The number of classes. When it is
                set, `n_classes` returns it and does not read
                ``dataset_metadata``.
            n_keypoints (int | None): The number of keypoints. When it
                is set, `n_keypoints` returns it and does not read
                ``dataset_metadata``.
            in_sizes (``Size | list[Size] | None``): The sizes of the
                attached inputs. When it is set, `in_sizes` returns it
                and does not read ``input_shapes``.
            remove_on_export (bool): When ``True``, the model skips the
                node in export mode, so the exported model does not
                contain the node.
            export_output_names (list[str] | None): The names of the
                node outputs in the exported model. See
                `export_output_names` for how the export uses them.
                ``None`` keeps the default names.
            attach_index (AttachIndexType | None): The output of the
                input node that the node reads. A value other than
                ``None`` replaces the class attribute and logs a
                warning. See `attach_index` for the accepted values.
            task_name (str | None): The dataset task of the node. It
                selects the classes and the keypoints in
                ``dataset_metadata``. ``None`` becomes ``""``.
            weights (``str | Literal["download", "yolo", "none"] | None``):
                The source or the initialization method of the weights.
                The variant metaclass calls ``__post_init__`` after the
                constructor. That step reads the value:

                - ``"download"`` calls `load_checkpoint`, which takes the
                  URL from `get_weights_url`.
                - A string that contains ``"://"`` calls
                  `load_checkpoint` with that URL.
                - Any other string goes to `initialize_weights` as the
                  method. A local checkpoint path also goes there, and
                  the base implementation does not load it.
                - ``None`` and ``""`` act as ``"none"``.

        Raises:
            AssertionError: When `attach_index` is ``None`` and
                ``forward`` has no parameters.
            IncompatibleError: When a property that the class body
                annotates has a value of a different type.

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
        self._weights = weights or "none"
        self._signature = get_signature(self.forward)

        self.current_epoch = 0

        self._check_type_overrides()

    def __post_init__(self) -> None:
        """Load or initialize the weights that ``weights`` selects.

        The variant metaclass calls it after the constructor. The value
        ``"download"`` calls `load_checkpoint` without a checkpoint. A
        value that contains ``"://"`` calls `load_checkpoint` with that
        URL. Any other value goes to `initialize_weights`.

        """
        if self._weights == "download":
            self.load_checkpoint()
        elif "://" in self._weights:
            self.load_checkpoint(ckpt=self._weights)
        else:
            self.initialize_weights(method=self._weights)

    def initialize_weights(
        self, method: Literal["yolo", "none"] | str | None = None
    ) -> None:
        """Initialize the weights of the node.

        The node calls it after construction with the ``weights``
        argument as ``method``, unless ``weights`` asks for a checkpoint.
        A subclass overrides it to initialize its own layers.

        The base implementation knows one method, ``"yolo"``. It sets
        ``eps`` to ``0.001`` and ``momentum`` to ``0.03`` in every
        `torch.nn.BatchNorm2d`. It also sets ``inplace`` to ``True`` in
        every ``Hardswish``, ``LeakyReLU``, ``ReLU``, ``ReLU6``, and
        ``SiLU`` activation. Other values change nothing.

        Args:
            method (``Literal["yolo", "none"] | str | None``): The name
                of the initialization method. ``None`` and ``"none"``
                change nothing.

        Example:
            The ``weights`` argument of the constructor selects the
            method.

            >>> from torch import Tensor, nn
            >>> from luxonis_train.nodes import BaseNode
            >>> class Node(BaseNode, register=False):
            ...     def __init__(self, **kwargs):
            ...         super().__init__(**kwargs)
            ...         self.bn = nn.BatchNorm2d(4)
            ...
            ...     def forward(self, x: Tensor) -> Tensor:
            ...         return self.bn(x)
            >>> Node().bn.eps
            1e-05
            >>> node = Node(weights="yolo")
            >>> node.bn.eps, node.bn.momentum
            (0.001, 0.03)

        """
        if method is None or method == "none":
            return

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
        """Return the default variant name and the variants of the node.

        A node with variants overrides this static method. A call such
        as ``Node(variant="n")`` selects a variant, and ``"default"``
        selects the default variant. The variant metaclass then passes
        the parameters of the variant to the constructor. An argument
        that the call gives explicitly replaces the variant parameter of
        the same name. A variant name that is not in the dictionary
        makes the metaclass raise ``ValueError``.

        Returns:
            ``tuple[str, dict[str, Kwargs]]``: The name of the default
            variant, and a dictionary that maps each variant name to its
            constructor keyword arguments.

        Raises:
            NotImplementedError: When the node has no variants. The base
                implementation always raises it.

        Example:
            >>> from torch import Tensor
            >>> from luxonis_train.nodes import BaseNode
            >>> class Node(BaseNode, register=False):
            ...     def __init__(self, width: int = 1, **kwargs):
            ...         super().__init__(**kwargs)
            ...         self.width = width
            ...
            ...     def forward(self, x: Tensor) -> Tensor:
            ...         return x
            ...
            ...     @staticmethod
            ...     def get_variants():
            ...         return "n", {"n": {"width": 8}, "s": {"width": 16}}
            >>> node = Node(variant="default")
            >>> node.variant, node.width
            ('n', 8)
            >>> Node(variant="s").width
            16
            >>> Node().width
            1

        """
        raise NotImplementedError

    @property
    def name(self) -> str:
        """The class name of the node.

        It is not the alias of the node in the config.

        """
        return self.__class__.__name__

    @property
    def variant(self) -> str:
        """The name of the variant that built the node.

        The variant metaclass sets it when a call selects a variant of a
        node that overrides `get_variants`. A ``"default"`` variant
        resolves to the name of the default variant.

        Raises:
            AttributeError: When no variant built the node. This occurs
                for a ``variant`` of ``"none"`` or ``None``, and for
                ``"default"`` on a node without variants.
            RuntimeError: When the variant name is ``None``.

        """
        if self._variant is None:
            raise RuntimeError(f"Variant was not set for node '{self.name}'.")
        return self._variant

    @property
    def n_keypoints(self) -> int:
        """The number of keypoints of the node task.

        The ``n_keypoints`` constructor argument comes first. Without
        it, the value comes from `dataset_metadata` for `task_name`. It
        is ``0`` when the dataset has no keypoints for that task.

        Raises:
            RuntimeError: When the constructor got neither
                ``n_keypoints`` nor ``dataset_metadata``.

        """
        if self._n_keypoints is not None:
            return self._n_keypoints

        return self.dataset_metadata.n_keypoints(self.task_name)

    @property
    def n_classes(self) -> int:
        """The number of classes of the node task.

        The ``n_classes`` constructor argument comes first. Without it,
        the value comes from `dataset_metadata` for `task_name`.

        Raises:
            RuntimeError: When the constructor got neither ``n_classes``
                nor ``dataset_metadata``.
            ValueError: When the dataset has no task named `task_name`.

        """
        if self._n_classes is not None:
            return self._n_classes

        return self.dataset_metadata.n_classes(self.task_name)

    @property
    def classes(self) -> bidict[str, int]:
        """The class indices of the node task, keyed by class name.

        The value always comes from `dataset_metadata` for `task_name`.
        The ``n_classes`` constructor argument does not change it. The
        value is a new ``bidict``. Its ``inverse`` maps the class indices
        to the class names.

        Raises:
            RuntimeError: When the constructor got no
                ``dataset_metadata``.
            ValueError: When the dataset has no task named `task_name`.

        """
        return self.dataset_metadata.classes(self.task_name)

    @property
    def class_names(self) -> list[str]:
        """The class names of the node task, sorted by class index.

        Raises:
            RuntimeError: When the constructor got no
                ``dataset_metadata``.
            ValueError: When the dataset has no task named `task_name`.

        """
        return [
            name for name, _ in sorted(self.classes.items(), key=itemgetter(1))
        ]

    @property
    def input_shapes(self) -> list[Packet[Size]]:
        """The shape packets of the node inputs, one for each input.

        The model takes the shapes from a run on zero tensors with a
        batch size of 2, so the shapes include the batch dimension.

        Raises:
            RuntimeError: When the constructor got no ``input_shapes``.

        """
        if self._input_shapes is None:
            raise self._non_set_error("input_shapes")
        return self._input_shapes

    @property
    def original_in_shape(self) -> Size:
        """The shape of the model input image, ``[C, H, W]``.

        The shape does not include the batch dimension.

        Raises:
            RuntimeError: When the constructor got no
                ``original_in_shape``.

        """
        if self._original_in_shape is None:
            raise self._non_set_error("original_in_shape")
        return self._original_in_shape

    @property
    def dataset_metadata(self) -> DatasetMetadata:
        """The metadata of the dataset.

        Raises:
            RuntimeError: When the constructor got no
                ``dataset_metadata``.

        """
        if self._dataset_metadata is None:
            raise RuntimeError(self._non_set_error("dataset_metadata"))
        return self._dataset_metadata

    @property
    def in_sizes(self) -> Size | list[Size]:
        """The sizes of the attached inputs.

        The property uses the first rule that applies:

        1. The ``in_sizes`` constructor argument, when it is set.
        2. The ``"features"`` entry of the only input packet.
        3. The only entry of that packet.
        4. The entries of that packet whose keys match the names of the
           ``forward`` parameters. Their sizes must be equal, and the
           property uses the first one.

        Rules 2 to 4 pass the entry through `get_attached`. The result is
        a single size for an integer `attach_index`, and a list of sizes
        for ``"all"`` or a range. A node with more than one input, or
        with shapes that the rules do not fit, must read `input_shapes`
        instead.

        Raises:
            RuntimeError: When ``input_shapes`` is missing or does not
                hold exactly one packet. Also when no key matches a
                ``forward`` parameter, or when the matching sizes differ.
                Also when `attach_index` is ``None`` and the entry is a
                list.
            ValueError: When `attach_index` does not fit the sizes.

        Example:
            >>> from torch import Size, Tensor
            >>> from luxonis_train.nodes import BaseNode
            >>> class Node(BaseNode, register=False):
            ...     def forward(self, x: list[Tensor]) -> list[Tensor]:
            ...         return x
            >>> shapes = [
            ...     {"features": [Size([2, 8, 64, 64]), Size([2, 16, 32, 32])]}
            ... ]
            >>> node = Node(input_shapes=shapes)
            >>> node.attach_index
            'all'
            >>> node.in_sizes
            [torch.Size([2, 8, 64, 64]), torch.Size([2, 16, 32, 32])]
            >>> node.in_channels, node.in_height, node.in_width
            ([8, 16], [64, 32], [64, 32])

        """
        if self._in_sizes is not None:
            return self._in_sizes

        if len(self.input_shapes) != 1:
            raise RuntimeError(
                f"Node '{self.name}' takes inputs from multiple "
                "preceding nodes, but the default implementation "
                "of `in_sizes` can only handle a single input nodes. "
                f"Please use `{self.name}.input_shapes` directly."
            )

        input_shapes = self.input_shapes[0]
        features = input_shapes.get("features")
        if features is not None:
            return self.get_attached(features)
        return self._infer_in_sizes_from_signature(input_shapes)

    def _infer_in_sizes_from_signature(
        self, input_shapes: Packet[Size]
    ) -> Size | list[Size]:
        if len(input_shapes) == 1:
            return self.get_attached(next(iter(input_shapes.values())))
        params = {}
        for name in self._signature:
            if name in input_shapes:
                params[name] = input_shapes[name]
        if not params:
            raise RuntimeError("Unable to determine the correct input shape.")
        if len(params) == 1:
            return self.get_attached(next(iter(params.values())))
        first = next(iter(params.values()))
        for value in params.values():
            if value != first:
                raise RuntimeError(
                    f"Node '{self.name}' requires multiple inputs, "
                    f"({list(params.keys())}) "
                    "but they are of different shapes. The default "
                    "implementation of `in_sizes` cannot be used. "
                    f"Please use `{self.name}.input_shapes` directly."
                )
        return self.get_attached(first)

    @property
    def in_channels(self) -> int | list[int]:
        """The number of channels of the attached inputs.

        It is the third dimension from the end of `in_sizes`, so a shape
        with or without the batch dimension gives the same value. A list
        of sizes gives a list of channel counts.

        Raises:
            RuntimeError: When `in_sizes` cannot find the input sizes.
            ValueError: When `attach_index` does not fit the sizes.

        """
        return self._get_nth_size(-3)

    @property
    def in_height(self) -> int | list[int]:
        """The height of the attached inputs.

        It is the second dimension from the end of `in_sizes`. A list of
        sizes gives a list of heights.

        Raises:
            RuntimeError: When `in_sizes` cannot find the input sizes.
            ValueError: When `attach_index` does not fit the sizes.

        """
        return self._get_nth_size(-2)

    @property
    def in_width(self) -> int | list[int]:
        """The width of the attached inputs.

        It is the last dimension of `in_sizes`. A list of sizes gives a
        list of widths.

        Raises:
            RuntimeError: When `in_sizes` cannot find the input sizes.
            ValueError: When `attach_index` does not fit the sizes.

        """
        return self._get_nth_size(-1)

    def get_weights_url(self) -> str:
        """Return the URL of the pretrained weights of the node.

        A node with pretrained weights overrides this method. The base
        implementation raises ``NotImplementedError``, which means that
        the node has no pretrained weights. `load_checkpoint` calls the
        method when it gets no checkpoint, for example for
        ``weights="download"``.

        The URL can contain these placeholders:

        - ``{github}`` becomes
          ``https://github.com/luxonis/luxonis-train/releases/download/v0.3.10-beta/``.
          The version is fixed. The installed version does not change
          it.
        - ``{github:v0.3.0}`` selects the release ``v0.3.0`` instead.
          The version needs three numbers and can have a suffix, as in
          ``v0.3.0-beta``.
        - ``{variant}`` becomes the name of the variant that built the
          node. The node must come from a variant.

        The file at the URL must hold the state dictionary of the node
        under the ``"state_dict"`` key. The keys of that dictionary must
        be the parameter and buffer names of the node.

        Returns:
            str: The URL of the checkpoint. It can contain the
            placeholders.

        Raises:
            NotImplementedError: When the node has no pretrained weights.

        """
        raise NotImplementedError

    def _get_weights_url(self) -> str | None:
        try:
            url = self.get_weights_url()
        except NotImplementedError:
            return None

        if "{variant}" in url:
            if self._variant is None:
                raise ValueError(
                    f"Attempting to get weights URL for '{self.name}' "
                    "node, but it uses the `{variant}` placeholder when "
                    "the node was not constructed from a variant."
                )
            url = url.replace("{variant}", self.variant)
        if match := re.search(
            r"\{github(?::(v[0-9]+\.[0-9]+\.[0-9]+(-\w+)?))?\}", url
        ):
            version = match.group(1) or "v0.3.10-beta"
            url = url.replace(
                match.group(0),
                "https://github.com/luxonis/"
                f"luxonis-train/releases/download/{version}/",
            )
        return url

    def load_checkpoint(
        self,
        ckpt: str | dict[str, Tensor] | None = None,
        *,
        strict: bool = True,
    ) -> None:
        """Load a checkpoint into the node.

        For a file, the method logs the path or URL. `safe_download`
        copies a remote file into the local cache first. When the
        download fails, the method logs a warning and leaves the weights
        unchanged. The method reads the file with `torch.load` on the
        CPU and with ``weights_only=False``. **Load only trusted files**,
        because the file can run code when it loads. After the load, the
        method logs an info message through the standard ``logging``
        module.

        Args:
            ckpt (``str | dict[str, Tensor] | None``): A state dictionary,
                or the local path or URL of a ``.ckpt`` file. The file
                must hold the state dictionary under the ``"state_dict"``
                key. ``None`` or ``""`` takes the URL from
                `get_weights_url`.
            strict (bool): Whether the keys of the state dictionary must
                match the keys of the node exactly. The value goes to
                `torch.nn.Module.load_state_dict`. With ``True``, that
                method raises ``RuntimeError`` when the keys differ.

        Raises:
            RuntimeError: When ``ckpt`` is an empty dictionary.
            ValueError: When ``ckpt`` is ``None`` and the node does not
                override `get_weights_url`.
            AttributeError: When ``ckpt`` is ``None``, the URL uses
                ``{variant}``, and no variant built the node.

        """
        if isinstance(ckpt, dict) and not ckpt:
            raise RuntimeError("Provided checkpoint dictionary is empty.")
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
            local_path = safe_download(ckpt)
            if local_path:
                # Load on the CPU. Lightning moves the node to the device.
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
        """Whether export mode is on.

        A node reads it in ``forward`` to return the outputs of the
        exported model. An assignment calls `set_export_mode`.

        """
        return self._export

    @export.setter
    def export(self, mode: bool) -> None:
        """Switch export mode on or off with `set_export_mode`."""
        self.set_export_mode(mode)

    def set_export_mode(self, /, mode: bool) -> None:
        """Switch export mode on or off.

        The method sets `export`. Then it visits the node and all its
        submodules. With ``True``, it calls ``reparameterize`` on each
        `Reparameterizable` module. With ``False``, it calls ``restore``
        on each of them. It logs every call at the debug level.

        Args:
            mode (bool): ``True`` to switch export mode on, ``False`` to
                switch it off.

        """
        self._export = mode

        for name, module in self.named_modules():
            if isinstance(module, Reparameterizable):
                if mode:
                    logger.debug(f"Reparameterizing '{name}' in '{self.name}'")
                    module.reparameterize()
                else:
                    logger.debug(
                        f"Restoring reparameterized '{name}' in '{self.name}'"
                    )
                    module.restore()

    @property
    def remove_on_export(self) -> bool:
        """Whether the model skips the node in export mode.

        The exported model then does not contain the node.

        """
        return self._remove_on_export

    @property
    def export_output_names(self) -> list[str] | None:
        """The names of the node outputs in the exported model.

        The base implementation returns the ``export_output_names``
        constructor argument. ``None`` keeps the default names.

        The ONNX export uses the names only when their number matches
        the number of node outputs. Otherwise, it logs a warning and
        keeps the default names. The NN Archive of a head lists the
        names as the head outputs.

        """
        return self._export_output_names

    @abstractmethod
    def forward(
        self,
        inputs: Tensor | list[Tensor] | Packet[Tensor] | list[Packet[Tensor]],
    ) -> Tensor | list[Tensor] | Packet[Tensor]:
        """Compute the outputs of the node.

        A subclass must implement it. `run` reads the name and the type
        annotation of each parameter to decide what the parameter gets.
        Annotate each parameter with one of the four types of
        ``inputs``. See `run` for the rules.

        Args:
            inputs (``Tensor | list[Tensor] | Packet[Tensor] | list[Packet[Tensor]]``):
                The input of the node. An implementation can rename the
                parameter and add more parameters.

        Returns:
            ``Tensor | list[Tensor] | Packet[Tensor]``: The outputs of the
            node. `run` puts a tensor or a list of tensors into a packet.

        """
        ...

    def run(self, inputs: list[Packet[Tensor]]) -> Packet[Tensor]:
        """Run the node on the packets of its inputs.

        The method gives each ``forward`` parameter a value, calls the
        node, and puts the result into a packet. The type annotation and
        the name of a parameter decide its value:

        - A ``list[Packet[Tensor]]`` parameter gets all input packets. It
          must be the only parameter.
        - A ``Packet[Tensor]`` parameter gets the input packet at the
          position of the parameter.
        - A ``Tensor`` or ``list[Tensor]`` parameter can have the name
          ``x``, ``y``, or ``z``, or a name that starts with ``input``.
          It then gets the ``"features"`` entry of one input packet.
          ``x``, ``y``, and ``z`` read the packets 0, 1, and 2. A number
          after ``input`` or ``inputs``, as in ``input_1``, selects that
          packet. Other names read packet 0. A list entry goes through
          `get_attached`.
        - A ``Tensor`` or ``list[Tensor]`` parameter with another name
          gets the entry with the same key, unchanged. Special case: no
          packet has the key, the node has one input packet, and
          ``forward`` has one parameter. Then the parameter gets the
          first entry of that packet through `get_attached`, and the
          method logs a warning.

        ``forward`` can return a packet, a tensor, or a list of tensors.
        The method puts a tensor or a list under the key
        ``task.main_output``, or under ``"features"`` when `task` is
        ``None``.

        Args:
            inputs (``list[Packet[Tensor]]``): One packet for each input
                of the node, in the order of the inputs.

        Returns:
            ``Packet[Tensor]``: The outputs of the node, for example
            ``{"features": [feature_map_1, feature_map_2]}``.

        Raises:
            TypeError: When a ``forward`` parameter has an annotation
                other than the four types above. The call to ``forward``
                also raises it when a required parameter gets no value.
            RuntimeError: When a ``list[Packet[Tensor]]`` parameter is
                not the only parameter, or when the input packets are too
                few. Also when a parameter that reads the ``"features"``
                entry gets a packet without that key. Also when an entry has the wrong type, or when two
                packets have the same key. Also when `get_attached` gets a list and
                `attach_index` is ``None``.
            ValueError: When ``forward`` returns a value of another type.
                Also when `attach_index` does not fit an entry that goes
                through `get_attached`.

        Example:
            >>> import torch
            >>> from torch import Tensor
            >>> from luxonis_train.nodes import BaseNode
            >>> class Add(BaseNode, register=False):
            ...     attach_index = -1
            ...
            ...     def forward(self, x: Tensor, y: Tensor) -> Tensor:
            ...         return x + y
            >>> packets = [
            ...     {"features": [torch.ones(2)]},
            ...     {"features": [torch.ones(2)]},
            ... ]
            >>> Add().run(packets)["features"].tolist()
            [2.0, 2.0]

        """
        kwargs: dict[str, _ForwardInput] = {}
        for i, (name, param) in enumerate(self._signature.items()):
            self._resolve_forward_param(i, name, param, inputs, kwargs)

        outputs = self(**kwargs)
        return self._normalize_output(outputs)

    def _resolve_forward_param(
        self,
        i: int,
        name: str,
        param: inspect.Parameter,
        inputs: list[Packet[Tensor]],
        kwargs: dict[str, _ForwardInput],
    ) -> None:
        if param.annotation == list[Packet[Tensor]]:
            if len(self._signature) != 1:
                raise RuntimeError(
                    f"Node '{self.name}' has a parameter '{name}' "
                    "of type `list[Packet[Tensor]]`, but it is not the "
                    "only parameter of the `forward` method. This is not "
                    "supported."
                )
            kwargs[name] = inputs
        elif param.annotation == Packet[Tensor]:
            if i >= len(inputs):
                raise RuntimeError(
                    f"Node '{self.name}' expects at least {i + 1} inputs, "
                    f"but received only {len(inputs)}."
                )
            kwargs[name] = inputs[i]
        elif param.annotation == list[Tensor] or param.annotation == Tensor:
            self._resolve_tensor_param(name, param, inputs, kwargs)
        else:
            raise TypeError(
                f"Node '{self.name}' has an unsupported type "
                f"`{param.annotation}` for parameter `{name}`. "
                "Supported types are Tensor, list of Tensors and "
                "Packet of Tensors."
            )

    def _resolve_tensor_param(
        self,
        name: str,
        param: inspect.Parameter,
        inputs: list[Packet[Tensor]],
        kwargs: dict[str, _ForwardInput],
    ) -> None:
        if (match := re.match(r"inputs?_?(\d+)?", name)) or name in "xyz":
            kwargs[name] = self._resolve_indexed_tensor(
                name, param, inputs, match
            )
        else:
            self._resolve_named_tensor(name, param, inputs, kwargs)

    def _resolve_indexed_tensor(
        self,
        name: str,
        param: inspect.Parameter,
        inputs: list[Packet[Tensor]],
        match: re.Match | None,
    ) -> Tensor | list[Tensor]:
        input_name = "features"
        if name in "xyz":
            idx = "xyz".index(name)
        elif match and match.group(1):
            idx = int(match.group(1))
        else:
            idx = 0

        if idx >= len(inputs):
            raise RuntimeError(
                f"Node '{self.name}' expects at least {idx + 1} inputs, "
                f"but received only {len(inputs)}."
            )
        packet = inputs[idx]
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
            return value
        return self.get_attached(value)

    def _resolve_named_tensor(
        self,
        name: str,
        param: inspect.Parameter,
        inputs: list[Packet[Tensor]],
        kwargs: dict[str, _ForwardInput],
    ) -> None:
        prev_kwargs_len = len(kwargs)

        for inp in inputs:
            if name in inp:
                if not check_type(inp[name], param.annotation):
                    raise RuntimeError(
                        f"Node '{self.name}' expects an input with key "
                        f"'{name}' to be of type `{param.annotation}`, "
                        f"but got `{type(inp[name])}` instead."
                    )
                if name in kwargs:
                    raise RuntimeError(
                        f"Node '{self.name}' requires an input with key "
                        f"'{name}', but it was found in multiple input packets."
                    )
                kwargs[name] = inp[name]
        if (
            len(kwargs) == prev_kwargs_len
            and len(inputs) == len(self._signature) == 1
            and name not in inputs[0]
        ):
            key_name = next(iter(inputs[0]))
            kwargs[name] = self.get_attached(next(iter(inputs[0].values())))

            logger.warning(
                f"Non-standard parameter name '{name}' used in `{self.name}.forward`. "
                f"The node expects a single argument of type `{param.annotation}` "
                f"and it got a single input packet with a single key '{key_name}'. "
                "Assuming the input corresponds to that parameter. "
                "If this is incorrect, please double check the parameter name or "
                "the input packets."
            )

    def _normalize_output(self, outputs: object) -> Packet[Tensor]:
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
    """The element type of `get_attached`: ``Tensor`` or ``Size``."""

    def get_attached(self, value: list[T] | T) -> list[T] | T:
        """Select the elements of a list that `attach_index` names.

        A value that is not a list passes unchanged. The index must then
        be ``None``, ``-1``, or ``0``. For a list, the index selects:

        - With ``"all"``, the whole list.
        - With an integer, one element. A negative index counts from the
          end.
        - With a pair ``(i, j)`` or a triple ``(i, j, k)``, a slice from
          ``i`` to ``j`` with the step ``k``. The slice includes ``i`` and
          leaves out ``j``, as in Python. A negative index counts from
          the end. The default step is ``-1`` when ``i > j`` and the two
          indices are both negative or both non-negative. Otherwise, it
          is ``1``.

        **Exception:** when ``i`` and ``j`` are both negative and
        ``i < j``, the range leaves out ``i`` and includes ``j``. The step
        is then always ``1``. Thus ``(-3, -1)`` selects the last two
        elements, not the Python slice ``[-3:-1]``.

        Args:
            value (``list[T] | T``): A list of tensors or sizes, or a
                single tensor or size.

        Returns:
            ``list[T] | T``: One element for an integer index, a list for
            ``"all"`` or a tuple, or ``value`` itself when it is not a
            list.

        Raises:
            ValueError: When ``value`` is not a list and the index is not
                ``None``, ``-1``, or ``0``. Also when an integer index is
                ``len(value)`` or larger.
            RuntimeError: When ``value`` is a list and the index is
                ``None``.

        Example:
            >>> from torch import Tensor
            >>> from luxonis_train.nodes import BaseNode
            >>> class Node(BaseNode, register=False):
            ...     def forward(self, x: Tensor) -> Tensor:
            ...         return x
            >>> node = Node()
            >>> node.get_attached([1, 2, 3, 4, 5])
            5
            >>> node.attach_index = (1, -1)
            >>> node.get_attached([1, 2, 3, 4, 5])
            [2, 3, 4]
            >>> node.attach_index = (-3, -1)
            >>> node.get_attached([1, 2, 3, 4, 5])
            [4, 5]

        """
        if not isinstance(value, list):
            if self.attach_index not in (None, -1, 0):
                raise ValueError(
                    f"Attach index for node '{self.name}' is set to "
                    f"'{self.attach_index}', but the input is not a list. "
                    "Only attach indices of None, -1 or 0 are valid in this case."
                )
            return value

        length = len(value)
        match self.attach_index:
            case "all":
                return value
            case int(i):
                if i < 0:
                    i += length
                if i >= length:
                    raise ValueError(
                        f"Attach index {i} is out of range "
                        f"for list of length {length}."
                    )
                return value[i]
            case (int(i), int(j)):
                return value[BaseNode._normalize_attach_slice(i, j, length)]
            case (int(i), int(j), int(k)):
                return value[BaseNode._normalize_attach_slice(i, j, length, k)]
            case None:
                raise RuntimeError(self._missing_attach_index_message())

    @staticmethod
    def _normalize_attach_slice(
        i: int, j: int, length: int, k: int | None = None
    ) -> slice:
        if i < 0 and j < 0:
            return BaseNode._both_negative_slice(i, j, length, k)
        if i < 0:
            return slice(length + i, j, k or 1)
        if j < 0:
            return slice(i, length + j, k or 1)
        if i > j:
            return slice(i, j, k or -1)
        return slice(i, j, k or 1)

    @staticmethod
    def _both_negative_slice(
        i: int, j: int, length: int, k: int | None
    ) -> slice:
        if i < j:
            return slice(max(length + i + 1, 0), length + j + 1, 1)
        return slice(length + i, length + j, (k or -1) if i > j else 1)

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
            "define it as a class attribute, or provide proper "
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
