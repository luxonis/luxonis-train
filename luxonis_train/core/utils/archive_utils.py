"""The NN Archive entries of an exported model and of its heads.

`LuxonisModel.archive` builds the ``inputs``, the ``outputs``, and the
``heads`` of the archive config with these functions.
`LuxonisModel.export` also reads the inputs and the outputs for the
``modelconverter`` config that it writes next to the ONNX file.

"""

from pathlib import Path
from typing import TypedDict

import onnx
from luxonis_ml.nn_archive.config_building_blocks import DataType
from onnx.onnx_pb import TensorProto

from luxonis_train.lightning import LuxonisLightningModule
from luxonis_train.nodes import BaseHead


class ArchiveMetadataDict(TypedDict):
    """The shape and the data type of one input or output of a model.

    Attributes:
        shape (list[int]): The dimensions of the tensor. A dimension
            without a fixed size in the model file is ``0``.
        dtype (``DataType``): The data type of the tensor, one of
            ``int8``, ``int32``, ``uint8``, ``float32``, and
            ``float16``.

    """

    shape: list[int]
    dtype: DataType


def get_inputs(path: Path) -> dict[str, ArchiveMetadataDict]:
    """Read the inputs of an exported model file.

    The function supports only ONNX files. It reads every entry of
    ``graph.input`` of the model.

    Args:
        path (``Path``): The model file, with the suffix ``.onnx``.

    Returns:
        ``dict[str, ArchiveMetadataDict]``: The shape and the data type
        of each input, keyed by input name, in graph order.

    Raises:
        NotImplementedError: When the suffix of ``path`` is not
            ``.onnx``.
        ValueError: When the file does not load as an ONNX model, or
            when an input has an unsupported data type.

    """
    if path.suffix == ".onnx":
        return _get_onnx_inputs(path)
    raise NotImplementedError(
        f"Missing input reading function for {path.suffix} models."
    )


def get_outputs(path: Path) -> dict[str, ArchiveMetadataDict]:
    """Read the outputs of an exported model file.

    The function supports only ONNX files. It reads every entry of
    ``graph.output`` of the model.

    Args:
        path (``Path``): The model file, with the suffix ``.onnx``.

    Returns:
        ``dict[str, ArchiveMetadataDict]``: The shape and the data type
        of each output, keyed by output name, in graph order.

    Raises:
        NotImplementedError: When the suffix of ``path`` is not
            ``.onnx``.
        ValueError: When the file does not load as an ONNX model, or
            when an output has an unsupported data type.

    """
    if path.suffix == ".onnx":
        return _get_onnx_outputs(path)
    raise NotImplementedError(
        f"Missing input reading function for {path.suffix} models."
    )


def _from_onnx_dtype(dtype: int) -> DataType:
    dtype_map: dict[int, str] = {
        TensorProto.INT8: "int8",
        TensorProto.INT32: "int32",
        TensorProto.UINT8: "uint8",
        TensorProto.FLOAT: "float32",
        TensorProto.FLOAT16: "float16",
    }
    if dtype not in dtype_map:  # pragma: no cover
        raise ValueError(f"Unsupported ONNX data type: `{dtype}`")

    return DataType(dtype_map[dtype])


def _load_onnx_model(onnx_path: Path) -> onnx.ModelProto:
    try:
        return onnx.load(str(onnx_path))
    except Exception as e:  # pragma: no cover
        raise ValueError(f"Failed to load ONNX model: `{onnx_path}`") from e


def _get_onnx_outputs(onnx_path: Path) -> dict[str, ArchiveMetadataDict]:
    model = _load_onnx_model(onnx_path)
    outputs: dict[str, ArchiveMetadataDict] = {}

    for output in model.graph.output:
        shape = [dim.dim_value for dim in output.type.tensor_type.shape.dim]
        outputs[output.name] = {
            "shape": shape,
            "dtype": _from_onnx_dtype(output.type.tensor_type.elem_type),
        }

    return outputs


def _get_onnx_inputs(onnx_path: Path) -> dict[str, ArchiveMetadataDict]:
    model = _load_onnx_model(onnx_path)

    inputs: dict[str, ArchiveMetadataDict] = {}

    for inp in model.graph.input:
        shape = [dim.dim_value for dim in inp.type.tensor_type.shape.dim]
        inputs[inp.name] = {
            "shape": shape,
            "dtype": _from_onnx_dtype(inp.type.tensor_type.elem_type),
        }

    return inputs


def _get_head_outputs(outputs: list[dict], head_name: str) -> list[str]:
    """Select the names of the model outputs that belong to a head.

    An output name with four parts separated by ``/`` matches when its
    second part is ``head_name``. Any other output name matches when it
    is ``head_name`` itself.

    Args:
        outputs (list[dict]): The outputs of the NN Archive config. The
            function reads the ``"name"`` key of each output.
        head_name (str): The name of the head node, such as
            ``"EfficientBBoxHead"``, or its alias.

    Returns:
        list[str]: The matching output names, in the order of
        ``outputs``.

    """
    output_names = []
    for output in outputs:
        try:
            _, name, _, _ = output["name"].split("/")
        except ValueError:
            name = output["name"]
        if name == head_name:
            output_names.append(output["name"])

    return output_names


def get_head_configs(
    lightning_module: LuxonisLightningModule, outputs: list[dict]
) -> list[dict]:
    """Build the ``heads`` entries of the NN Archive config.

    The function visits the nodes of ``lightning_module`` in build
    order. It skips a node that is not a `BaseHead`, and a head with
    ``remove_on_export`` set. For each other head, it starts from
    `BaseHead.get_head_config` and adds two keys:

    - ``"name"``: the name of the node. When an earlier head already
      took that name, the function appends ``_<n>``, where ``<n>`` is
      the number of names taken so far.
    - ``"outputs"``: the `BaseNode.export_output_names` of the head.
      When they are ``None`` or empty, the function selects the names
      in ``outputs`` that belong to the node name.

    Args:
        lightning_module (LuxonisLightningModule): The module whose
            heads the archive describes.
        outputs (list[dict]): The outputs of the NN Archive config, each
            with a ``"name"`` key.

    Returns:
        list[dict]: One config dictionary for each exported head, with
        the keys ``"parser"``, ``"metadata"``, ``"name"``, and
        ``"outputs"``.

    """
    head_configs = []
    head_names = set()

    for node_name, node_wrapper in lightning_module.nodes.items():
        node = node_wrapper.module
        if not isinstance(node, BaseHead) or node.remove_on_export:
            continue
        head_config = node.get_head_config()
        head_name = (
            node_name
            if node_name not in head_names
            else f"{node_name}_{len(head_names)}"
        )
        head_names.add(head_name)

        head_outputs = node.export_output_names or _get_head_outputs(
            outputs, node_name
        )
        head_config.update({"name": head_name, "outputs": head_outputs})

        head_configs.append(head_config)

    return head_configs
