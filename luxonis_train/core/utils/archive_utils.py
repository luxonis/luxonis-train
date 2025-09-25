from pathlib import Path
from typing import TypedDict

import onnx
from luxonis_ml.nn_archive.config_building_blocks import DataType
from onnx.onnx_pb import TensorProto

from luxonis_train.lightning import LuxonisLightningModule
from luxonis_train.nodes import BaseHead


class ArchiveMetadataDict(TypedDict):
    shape: list[int]
    dtype: DataType


def get_inputs(path: Path) -> dict[str, ArchiveMetadataDict]:
    """Get inputs of a model executable.

    @type path: Path
    @param path: Path to model executable file.
    """
    if path.suffix == ".onnx":
        return _get_onnx_inputs(path)
    raise NotImplementedError(
        f"Missing input reading function for {path.suffix} models."
    )


def get_outputs(path: Path) -> dict[str, ArchiveMetadataDict]:
    """Get outputs of a model executable.

    @type path: Path
    @param path: Path to model executable file.
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
    """Get model outputs in a head-specific format.

    @type outputs: list[dict]
    @param outputs: List of NN Archive outputs.
    @type head_name: str
    @param head_name: Type of the head (e.g. 'EfficientBBoxHead') or its
        custom alias.
    @rtype: list[str]
    @return: List of output names.
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
    """Get model heads.

    @type lightning_module: LuxonisLightningModule
    @param lightning_module: Lightning module.
    @type outputs: list[dict]
    @param outputs: List of NN Archive outputs.
    @rtype: list[dict]
    @return: List of head configurations.
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
