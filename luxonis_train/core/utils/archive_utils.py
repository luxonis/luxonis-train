import logging
from collections import defaultdict
from pathlib import Path
from typing import TypedDict

import onnx
from luxonis_ml.nn_archive.config_building_blocks import (
    DataType,
)
from onnx.onnx_pb import TensorProto

from luxonis_train.config import Config
from luxonis_train.nodes.base_node import BaseNode
from luxonis_train.nodes.enums.head_categorization import (
    ImplementedHeads,
    ImplementedHeadsIsSoxtmaxed,
)

logger = logging.getLogger(__name__)


class MetadataDict(TypedDict):
    shape: list[int]
    dtype: DataType


def get_inputs(path: Path) -> dict[str, MetadataDict]:
    """Get inputs of a model executable.

    @type path: Path
    @param path: Path to model executable file.
    """

    if path.suffix == ".onnx":
        return _get_onnx_inputs(path)
    else:
        raise NotImplementedError(
            f"Missing input reading function for {path.suffix} models."
        )


def get_outputs(path: Path) -> dict[str, MetadataDict]:
    """Get outputs of a model executable.

    @type path: Path
    @param path: Path to model executable file.
    """

    if path.suffix == ".onnx":
        return _get_onnx_outputs(path)
    else:
        raise NotImplementedError(
            f"Missing input reading function for {path.suffix} models."
        )


def _from_onnx_dtype(dtype: int) -> DataType:
    dtype_map = {
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


def _get_onnx_outputs(onnx_path: Path) -> dict[str, MetadataDict]:
    model = _load_onnx_model(onnx_path)
    outputs: dict[str, MetadataDict] = defaultdict(dict)  # type: ignore

    for output in model.graph.output:
        shape = [dim.dim_value for dim in output.type.tensor_type.shape.dim]
        outputs[output.name]["shape"] = shape
        outputs[output.name]["dtype"] = _from_onnx_dtype(
            output.type.tensor_type.elem_type
        )

    return outputs


def _get_onnx_inputs(onnx_path: Path) -> dict[str, MetadataDict]:
    model = _load_onnx_model(onnx_path)

    inputs: dict[str, MetadataDict] = defaultdict(dict)  # type: ignore

    for inp in model.graph.input:
        shape = [dim.dim_value for dim in inp.type.tensor_type.shape.dim]
        inputs[inp.name]["shape"] = shape
        inputs[inp.name]["dtype"] = _from_onnx_dtype(
            inp.type.tensor_type.elem_type
        )

    return inputs


def _get_classes(
    node_name: str, node_task: str | None, classes: dict[str, list[str]]
) -> list[str]:
    if not node_task:
        match node_name:
            case "ClassificationHead":
                node_task = "classification"
            case "EfficientBBoxHead":
                node_task = "boundingbox"
            case "SegmentationHead" | "BiSeNetHead" | "DDRNetSegmentationHead":
                node_task = "segmentation"
            case "EfficientKeypointBBoxHead":
                node_task = "keypoints"
            case _:  # pragma: no cover
                raise ValueError("Node does not map to a default task.")

    return classes.get(node_task, [])


def _get_head_specific_parameters(
    nodes: dict[str, BaseNode], head_name: str, head_alias: str
) -> dict:
    """Get parameters specific to head.

    @type nodes: dict[str, BaseNode]
    @param nodes: Dictionary of nodes.
    @type head_name: str
    @param head_name: Name of the head (e.g. 'EfficientBBoxHead').
    @type head_alias: str
    @param head_alias: Alias of the head (e.g. 'detection_head').
    """

    parameters = {}
    if head_name == "ClassificationHead":
        parameters["is_softmax"] = getattr(
            ImplementedHeadsIsSoxtmaxed, head_name
        ).value
    elif head_name == "EfficientBBoxHead":
        parameters["subtype"] = "yolov6"
        head_node = nodes[head_alias]
        parameters["iou_threshold"] = head_node.iou_thres
        parameters["conf_threshold"] = head_node.conf_thres
        parameters["max_det"] = head_node.max_det
    elif head_name in [
        "SegmentationHead",
        "BiSeNetHead",
        "DDRNetSegmentationHead",
    ]:
        parameters["is_softmax"] = getattr(
            ImplementedHeadsIsSoxtmaxed, head_name
        ).value
    elif head_name == "EfficientKeypointBBoxHead":
        # or appropriate subtype
        head_node = nodes[head_alias]
        parameters["iou_threshold"] = head_node.iou_thres
        parameters["conf_threshold"] = head_node.conf_thres
        parameters["max_det"] = head_node.max_det
        parameters["n_keypoints"] = head_node.n_keypoints
    else:  # pragma: no cover
        raise ValueError("Unknown head name")
    return parameters


def _get_head_outputs(
    outputs: list[dict], head_name: str, head_type: str
) -> list[str]:
    """Get model outputs in a head-specific format.

    @type outputs: list[dict]
    @param outputs: List of NN Archive outputs.
    @type head_name: str
    @param head_name: Type of the head (e.g. 'EfficientBBoxHead') or its
        custom alias.
    @type head_type: str
    @param head_name: Type of the head (e.g. 'EfficientBBoxHead').
    @rtype: list[str]
    @return: List of output names.
    """

    output_names = []
    for output in outputs:
        name = output["name"].split("/")[0]
        if name == head_name:
            output_names.append(output["name"])

    return output_names


def get_heads(
    cfg: Config,
    outputs: list[dict],
    class_dict: dict[str, list[str]],
    nodes: dict[str, BaseNode],
) -> list[dict]:
    """Get model heads.

    @type cfg: Config
    @param cfg: Configuration object.
    @type outputs: list[dict]
    @param outputs: List of model outputs.
    @type class_dict: dict[str, list[str]]
    @param class_dict: Dictionary of classes.
    @type nodes: dict[str, BaseNode]
    @param nodes: Dictionary of nodes.
    """
    heads = []
    head_names = set()
    for node in cfg.model.nodes:
        node_name = node.name
        node_alias = node.alias or node_name
        if "aux-segmentation" in node_alias:
            continue
        if node_alias in cfg.model.outputs:
            if node_name in ImplementedHeads.__members__:
                parser = getattr(ImplementedHeads, node_name).value
                task = node.task
                if isinstance(task, dict):
                    task = str(next(iter(task.values())))

                classes = _get_classes(node_name, task, class_dict)

                export_output_names = nodes[node_alias].export_output_names
                if export_output_names is not None:
                    head_outputs = export_output_names
                else:
                    head_outputs = _get_head_outputs(
                        outputs, node_alias, node_name
                    )

                if node_alias in head_names:
                    curr_head_name = f"{node_alias}_{len(head_names)}"  # add suffix if name is already present
                else:
                    curr_head_name = node_alias
                head_names.add(curr_head_name)
                head_dict = {
                    "name": curr_head_name,
                    "parser": parser,
                    "metadata": {
                        "classes": classes,
                        "n_classes": len(classes),
                    },
                    "outputs": head_outputs,
                }
                head_dict["metadata"].update(
                    _get_head_specific_parameters(nodes, node_name, node_alias)
                )
                heads.append(head_dict)
    return heads
