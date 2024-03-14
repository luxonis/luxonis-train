import os
from logging import getLogger
from pathlib import Path
from typing import Any

import onnx
from luxonis_ml.nn_archive.archive_generator import ArchiveGenerator
from luxonis_ml.nn_archive.config import CONFIG_VERSION
from luxonis_ml.utils import LuxonisFileSystem

from luxonis_train.models import LuxonisModel
from luxonis_train.nodes.enums.head_categorization import ImplementedHeads
from luxonis_train.utils.config import Config

from .core import Core

logger = getLogger(__name__)


class Archiver(Core):
    """Main API which is used to construct the NN archive out of a trainig config and
    model executables."""

    def __init__(
        self,
        cfg: str | dict[str, Any] | Config,
        opts: list[str] | tuple[str, ...] | dict[str, Any] | None = None,
    ):
        """Constructs a new Archiver instance.

        @type cfg: str | dict[str, Any] | Config
        @param cfg: Path to config file or config dict used to setup training.
        @type opts: list[str] | tuple[str, ...] | dict[str, Any] | None
        @param opts: Argument dict provided through command line,
            used for config overriding.
        """

        super().__init__(cfg, opts)

        self.lightning_module = LuxonisModel(
            cfg=self.cfg,
            dataset_metadata=self.dataset_metadata,
            save_dir=self.run_save_dir,
            input_shape=self.loader_train.input_shape,
        )

        self.model_name = self.cfg.model.name

        self.archive_name = self.cfg.archiver.archive_name
        archive_save_directory = Path(self.cfg.archiver.archive_save_directory)
        if not archive_save_directory.exists():
            logger.info(f"Creating archive directory {archive_save_directory}")
            archive_save_directory.mkdir(parents=True, exist_ok=True)
        self.archive_save_directory = str(archive_save_directory)

        self.inputs = []
        self.outputs = []
        self.heads = []

    def archive(self, executable_path: str):
        """Runs archiving.

        @type executable_path: str
        @param executable_path: Path to model executable file (e.g. ONNX model).
        """

        executable_fname = os.path.split(executable_path)[1]
        _, executable_suffix = os.path.splitext(executable_fname)
        self.archive_name += f"_{executable_suffix[1:]}"

        preprocessing = {  # TODO: keep preprocessing same for each input?
            "mean": self.cfg.trainer.preprocessing.normalize.params["mean"],
            "scale": self.cfg.trainer.preprocessing.normalize.params["std"],
            "reverse_channels": self.cfg.trainer.preprocessing.train_rgb,
            "interleaved_to_planar": False,  # TODO: make it modifiable?
        }

        inputs_dict = self._get_inputs(executable_path)
        for input_name in inputs_dict:
            self._add_input(
                name=input_name,
                dtype=inputs_dict[input_name]["dtype"],
                shape=inputs_dict[input_name]["shape"],
                preprocessing=preprocessing,
            )

        outputs_dict = self._get_outputs(executable_path)
        for output_name in outputs_dict:
            self._add_output(name=output_name, dtype=outputs_dict[output_name]["dtype"])

        heads_dict = self._get_heads(executable_path)
        for head_name in heads_dict:
            self._add_head(heads_dict[head_name])

        model = {
            "metadata": {
                "name": self.model_name,
                "path": executable_fname,
            },
            "inputs": self.inputs,
            "outputs": self.outputs,
            "heads": self.heads,
        }

        cfg_dict = {
            "config_version": CONFIG_VERSION.__args__[0],
            "model": model,
        }

        self.archive_path = ArchiveGenerator(
            archive_name=self.archive_name,
            save_path=self.archive_save_directory,
            cfg_dict=cfg_dict,
            executables_paths=[executable_path],  # TODO: what if more executables?
        ).make_archive()

        if self.cfg.archiver.upload_url is not None:
            self._upload()

        return self.archive_path

    def _get_inputs(self, executable_path: str):
        """Get inputs of a model executable.

        @type executable_path: str
        @param executable_path: Path to model executable file.
        """

        _, executable_suffix = os.path.splitext(executable_path)
        if executable_suffix == ".onnx":
            return self._get_onnx_inputs(executable_path)
        else:
            raise NotImplementedError(
                f"Missing input reading function for {executable_suffix} models."
            )

    def _get_onnx_inputs(self, executable_path: str):
        """Get inputs of an ONNX model executable.

        @type executable_path: str
        @param executable_path: Path to model executable file.
        """

        inputs_dict = {}
        model = onnx.load(executable_path)
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
            inputs_dict[input.name] = {"dtype": dtype, "shape": shape}
        return inputs_dict

    def _add_input(
        self,
        name: str,
        dtype: str,
        shape: list,
        preprocessing: dict,
        input_type: str = "image",
    ) -> None:
        """Add input to self.inputs.

        @type name: str
        @param name: Name of the input layer.
        @type dtype: str
        @param dtype: Data type of the input data (e.g., 'float32').
        @type shape: list
        @param shape: Shape of the input data as a list of integers (e.g. [H,W], [H,W,C], [BS,H,W,C], ...).
        @type preprocessing: dict
        @param preprocessing: Preprocessing steps applied to the input data.
        @type input_type: str
        @param input_type: Type of input data (e.g., 'image').
        """

        self.inputs.append(
            {
                "name": name,
                "dtype": dtype,
                "input_type": input_type,
                "shape": shape,
                "preprocessing": preprocessing,
            }
        )

    def _get_outputs(self, executable_path):
        """Get outputs of a model executable.

        @type executable_path: str
        @param executable_path: Path to model executable file.
        """

        _, executable_suffix = os.path.splitext(executable_path)
        if executable_suffix == ".onnx":
            return self._get_onnx_outputs(executable_path)
        else:
            raise NotImplementedError(
                f"Missing input reading function for {executable_suffix} models."
            )

    def _get_onnx_outputs(self, executable_path):
        """Get outputs of an ONNX model executable.

        @type executable_path: str
        @param executable_path: Path to model executable file.
        """

        outputs_dict = {}
        model = onnx.load(executable_path)
        for output in model.graph.output:
            tensor_type = output.type.tensor_type
            dtype_idx = tensor_type.elem_type
            dtype = str(onnx.helper.tensor_dtype_to_np_dtype(dtype_idx))
            outputs_dict[output.name] = {"dtype": dtype}
        return outputs_dict

    def _add_output(self, name: str, dtype: str) -> None:
        """Add output to self.outputs.

        @type name: str
        @param name: Name of the output layer.
        @type dtype: str
        @param dtype: Data type of the output data (e.g., 'float32').
        """

        self.outputs.append({"name": name, "dtype": dtype})

    def _get_classes(self, head_family):
        if head_family.startswith("Classification"):
            return self.dataset_metadata._classes["class"]
        elif head_family.startswith("Object"):
            return self.dataset_metadata._classes["boxes"]
        elif head_family.startswith("Segmentation"):
            return self.dataset_metadata._classes["segmentation"]
        elif head_family.startswith("Keypoint"):
            return self.dataset_metadata._classes["keypoints"]
        else:
            raise ValueError(
                f"No classes found for the specified head family ({head_family})"
            )

    def _get_head_specific_parameters(
        self, head_name, head_alias, executable_path
    ) -> dict:
        """Get parameters specific to head.

        @type head_name: str
        @param head_name: Name of the head (e.g. 'EfficientBBoxHead').
        @type head_alias: str
        @param head_alias: Alias of the head (e.g. 'detection_head').
        @type executable_path: str
        @param executable_path: Path to model executable file.
        """

        parameters = {}
        if head_name == "ClassificationHead":
            parameters["is_softmax"] = self._is_softmax(executable_path)  # TODO: test
        elif head_name == "EfficientBBoxHead":
            parameters["subtype"] = "yolov6"
            head_node = self.lightning_module._modules["nodes"][head_alias]
            parameters["iou_threshold"] = head_node.iou_thres
            parameters["conf_threshold"] = head_node.conf_thres
            parameters["max_det"] = head_node.max_det
            # head_outputs["n_keypoints"] # TODO: implement
            # head_outputs["n_prototypes"] # TODO: implement
            # head_outputs["prototype_output_name"] # TODO: implement
        elif head_name == "ObjectDetectionSSD":
            raise NotImplementedError
            # head_outputs["anchors"] # TODO: implement
        elif head_name == "SegmentationHead":
            parameters["is_softmax"] = self._is_softmax(executable_path)  # TODO: test
        elif head_name == "BiSeNetHead":
            parameters["is_softmax"] = self._is_softmax(executable_path)  # TODO: test
        elif head_name == "ImplicitKeypointBBoxHead":
            pass
        else:
            raise ValueError("Unknown head name")
        return parameters

    def _is_softmax(self, executable_path) -> bool:
        """Check if model output is softmaxed.

        @type executable_path: str
        @param executable_path: Path to model executable file.
        """

        _, executable_suffix = os.path.splitext(executable_path)
        if executable_suffix == ".onnx":
            return self._is_softmax_onnx(executable_path)
        else:
            raise NotImplementedError(
                f"Missing softmax checking function for {executable_suffix} models."
            )

    def _is_softmax_onnx(self, executable_path) -> bool:
        """Check if ONNX model output is softmaxed.

        @type executable_path: str
        @param executable_path: Path to model executable file.
        """

        model = onnx.load(executable_path)
        for node in model.graph.node:
            if node.op_type.lower() == "softmax":
                return True
        return False

    def _get_head_outputs(self, head_name) -> dict:
        """Get model outputs in a head-specific format.

        @type head_name: str
        @param head_name: Name of the head (e.g. 'EfficientBBoxHead').
        """

        head_outputs = {}
        if head_name == "ClassificationHead":
            head_outputs["predictions"] = self.outputs[0]["name"]  # TODO: test
        elif head_name == "EfficientBBoxHead":
            head_outputs["yolo_outputs"] = [
                output_dict["name"] for output_dict in self.outputs
            ]  # TODO: test
        elif head_name == "ObjectDetectionSSD":
            raise NotImplementedError  # TODO: boxes, scores
        elif head_name == "SegmentationHead":
            raise NotImplementedError  # TODO: predictions
        elif head_name == "BiSeNetHead":
            raise NotImplementedError
        elif head_name == "ImplicitKeypointBBoxHead":
            head_outputs["predictions"] = self.outputs[0]["name"]  # TODO: test
        else:
            raise ValueError("Unknown head name")
        return head_outputs

    def _get_heads(self, executable_path):
        """Get model heads.

        @type executable_path: str
        @param executable_path: Path to model executable file.
        """
        heads_dict = {}

        for node in self.cfg.model.nodes:
            node_name = node.name
            node_alias = node.alias
            # node_inputs = node.inputs
            if node_alias in self.lightning_module.outputs:
                if node_name in ImplementedHeads.__members__:
                    head_family = getattr(ImplementedHeads, node_name).value
                    classes = self._get_classes(head_family)
                    head_outputs = self._get_head_outputs(node_name)
                    head_dict = {
                        "family": head_family,
                        "outputs": head_outputs,
                        "classes": classes,
                        "n_classes": len(classes),
                    }
                    head_dict.update(
                        self._get_head_specific_parameters(
                            node_name, node_alias, executable_path
                        )
                    )
                    heads_dict[node_name] = head_dict
        return heads_dict

    def _add_head(self, head_metadata: dict) -> str:
        """Add head to self.heads.

        @type metadata: dict
        @param metadata: Parameters required by head to run postprocessing.
        """

        self.heads.append(head_metadata)

    def _upload(self):
        """Uploads the archive file to specified s3 bucket.

        @raises ValueError: If upload url was not specified in config file.
        """

        if self.cfg.archiver.upload_url is None:
            raise ValueError("Upload url must be specified in config file.")

        fs = LuxonisFileSystem(self.cfg.archiver.upload_url, allow_local=False)
        logger.info(f"Started Archive upload to {fs.full_path}...")

        fs.put_file(
            local_path=self.archive_path,
            remote_path=self.archive_name,
        )

        logger.info("Files upload finished")
