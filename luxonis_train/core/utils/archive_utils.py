from pathlib import Path

import onnx
from luxonis_ml.nn_archive.config_building_blocks import ObjectDetectionSubtypeYOLO

from luxonis_train.nodes.base_node import BaseNode
from luxonis_train.nodes.enums.head_categorization import (
    ImplementedHeads,
    ImplementedHeadsIsSoxtmaxed,
)
from luxonis_train.utils.config import Config


def get_inputs(path: Path):
    """Get inputs of a model executable.

    @type path: Path
    @param path: Path to model executable file.
    """

    if path.suffix == ".onnx":
        return _get_onnx_inputs(str(path))
    else:
        raise NotImplementedError(
            f"Missing input reading function for {path.suffix} models."
        )


def _get_onnx_inputs(path: str) -> dict:
    """Get inputs of an ONNX model executable.

    @type path: str
    @param path: Path to model executable file.
    """

    inputs_dict = {}
    model = onnx.load(path)
    for input in model.graph.input:
        tensor_type = input.type.tensor_type
        dtype_idx = tensor_type.elem_type
        dtype = str(onnx.helper.tensor_dtype_to_np_dtype(dtype_idx))
        shape = []
        for d in tensor_type.shape.dim:
            if d.HasField("dim_value"):
                shape.append(d.dim_value)
            else:
                raise ValueError("Unsupported input dimension identifier type")
        if shape[1] == 3:
            layout = "NCHW"
        elif shape[3] == 3:
            layout = "NHWC"
        else:
            raise ValueError("Unknown input layout")
        inputs_dict[input.name] = {"dtype": dtype, "shape": shape, "layout": layout}
    return inputs_dict


def get_outputs(path: Path) -> dict:
    """Get outputs of a model executable.

    @type path: Path
    @param path: Path to model executable file.
    """

    if path.suffix == ".onnx":
        return _get_onnx_outputs(str(path))
    else:
        raise NotImplementedError(
            f"Missing input reading function for {path.suffix} models."
        )


def _get_onnx_outputs(path: str) -> dict:
    """Get outputs of an ONNX model executable.

    @type executable_path: str
    @param executable_path: Path to model executable file.
    """

    outputs_dict = {}
    model = onnx.load(path)
    for output in model.graph.output:
        tensor_type = output.type.tensor_type
        dtype_idx = tensor_type.elem_type
        dtype = str(onnx.helper.tensor_dtype_to_np_dtype(dtype_idx))
        outputs_dict[output.name] = {"dtype": dtype}
    return outputs_dict


def _get_classes(
    node_name: str, node_task: str | None, classes: dict[str, list[str]]
) -> list[str]:
    if not node_task:
        match node_name:
            case "ClassificationHead":
                node_task = "classification"
            case "EfficientBBoxHead":
                node_task = "boundingbox"
            case "SegmentationHead" | "BiSeNetHead":
                node_task = "segmentation"
            case "ImplicitKeypointBBoxHead" | "EfficientKeypointBBoxHead":
                node_task = "keypoints"
            case _:
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
        parameters["is_softmax"] = getattr(ImplementedHeadsIsSoxtmaxed, head_name).value
    elif head_name == "EfficientBBoxHead":
        parameters["subtype"] = ObjectDetectionSubtypeYOLO.YOLOv6.value
        head_node = nodes[head_alias]
        parameters["iou_threshold"] = head_node.iou_thres
        parameters["conf_threshold"] = head_node.conf_thres
        parameters["max_det"] = head_node.max_det
    elif head_name in ["SegmentationHead", "BiSeNetHead"]:
        parameters["is_softmax"] = getattr(ImplementedHeadsIsSoxtmaxed, head_name).value
    elif head_name == "ImplicitKeypointBBoxHead":
        parameters["subtype"] = ObjectDetectionSubtypeYOLO.YOLOv7.value
        head_node = nodes[head_alias]
        parameters["iou_threshold"] = head_node.iou_thres
        parameters["conf_threshold"] = head_node.conf_thres
        parameters["max_det"] = head_node.max_det
        parameters["n_keypoints"] = head_node.n_keypoints
        parameters["anchors"] = head_node.anchors.tolist()
    elif head_name == "EfficientKeypointBBoxHead":
        # or appropriate subtype
        head_node = nodes[head_alias]
        parameters["iou_threshold"] = head_node.iou_thres
        parameters["conf_threshold"] = head_node.conf_thres
        parameters["max_det"] = head_node.max_det
        parameters["n_keypoints"] = head_node.n_keypoints
    else:
        raise ValueError("Unknown head name")
    return parameters


def _get_head_outputs(outputs: list[dict], head_name: str) -> list[str]:
    """Get model outputs in a head-specific format.

    @type head_name: str
    @param head_name: Name of the head (e.g. 'EfficientBBoxHead').
    @rtype: list[str]
    @return: List of output names.
    """

    if head_name == "ClassificationHead":
        return [outputs[0]["name"]]
    elif head_name == "EfficientBBoxHead":
        return [output["name"] for output in outputs]
    elif head_name in ["SegmentationHead", "BiSeNetHead"]:
        return [outputs[0]["name"]]
    elif head_name == "ImplicitKeypointBBoxHead":
        return [outputs[0]["name"]]
    elif head_name == "EfficientKeypointBBoxHead":
        return [outputs[0]["name"]]
    else:
        raise ValueError("Unknown head name")


def get_heads(
    cfg: Config,
    outputs: list[dict],
    class_dict: dict[str, list[str]],
    nodes: dict[str, BaseNode],
) -> dict[str, dict]:
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
    heads_dict = {}

    for node in cfg.model.nodes:
        node_name = node.name
        node_alias = node.alias or node_name
        if node_alias in cfg.model.outputs:
            if node_name in ImplementedHeads.__members__:
                parser = getattr(ImplementedHeads, node_name).value
                task = node.task
                if isinstance(task, dict):
                    task = str(next(iter(task)))

                classes = _get_classes(node_name, task, class_dict)
                head_outputs = _get_head_outputs(outputs, node_name)
                head_dict = {
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
                heads_dict[node_name] = head_dict
    return heads_dict
