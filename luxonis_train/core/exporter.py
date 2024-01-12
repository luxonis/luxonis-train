import os
import tempfile
from logging import getLogger
from pathlib import Path
from typing import Any

import onnx
import yaml
from luxonis_ml.utils import LuxonisFileSystem
from torch import Size

from luxonis_train.models import LuxonisModel
from luxonis_train.utils.config import Config

from .core import Core

logger = getLogger(__name__)


class Exporter(Core):
    """Main API which is used to create the model, setup pytorch lightning environment
    and perform training based on provided arguments and config."""

    def __init__(
        self,
        cfg: str | dict[str, Any] | Config,
        opts: list[str] | tuple[str, ...] | dict[str, Any] | None = None,
    ):
        """Constructs a new Exporter instance.

        @type cfg: str | dict[str, Any] | Config
        @param cfg: Path to config file or config dict used to setup training.

        @type opts: list[str] | tuple[str, ...] | dict[str, Any] | None
        @param opts: Argument dict provided through command line,
            used for config overriding.
        """

        super().__init__(cfg, opts)

        input_shape = self.cfg.exporter.input_shape
        if self.cfg.model.weights is None:
            raise ValueError(
                "Model weights must be specified in config file for export."
            )
        self.local_path = self.cfg.model.weights
        if input_shape is None:
            self.input_shape = self.loader_val.input_shape
        else:
            self.input_shape = Size(input_shape)

        export_path = (
            Path(self.cfg.exporter.export_save_directory)
            / self.cfg.exporter.export_model_name
        )

        if not export_path.parent.exists():
            logger.info(f"Creating export directory {export_path.parent}")
            export_path.parent.mkdir(parents=True, exist_ok=True)
        self.export_path = str(export_path)

        normalize_params = self.cfg.trainer.preprocessing.normalize.params
        if self.cfg.exporter.scale_values is not None:
            self.scale_values = self.cfg.exporter.scale_values
        else:
            self.scale_values = normalize_params.get("std", None)
            if self.scale_values:
                self.scale_values = (
                    [i * 255 for i in self.scale_values]
                    if isinstance(self.scale_values, list)
                    else self.scale_values * 255
                )

        if self.cfg.exporter.mean_values is not None:
            self.mean_values = self.cfg.exporter.mean_values
        else:
            self.mean_values = normalize_params.get("mean", None)
            if self.mean_values:
                self.mean_values = (
                    [i * 255 for i in self.mean_values]
                    if isinstance(self.mean_values, list)
                    else self.mean_values * 255
                )

        self.lightning_module = LuxonisModel(
            cfg=self.cfg,
            save_dir=self.run_save_dir,
            input_shape=self.input_shape,
            dataset_metadata=self.dataset_metadata,
        )

    def _get_modelconverter_config(self, onnx_path: str) -> dict[str, Any]:
        """Generates export config from input config that is compatible with Luxonis
        modelconverter tool.

        @type onnx_path: str
        @param onnx_path: Path to .onnx model
        @rtype: dict[str, Any]
        @return: Export config.
        """
        return {
            "input_model": onnx_path,
            "scale_values": self.scale_values,
            "mean_values": self.mean_values,
            "reverse_input_channels": self.cfg.exporter.reverse_input_channels,
            "use_bgr": not self.cfg.trainer.preprocessing.train_rgb,
            "input_shape": list(self.input_shape),
            "data_type": self.cfg.exporter.data_type,
            "output": [{"name": name} for name in self.output_names],
            "meta": {"description": self.cfg.model.name},
        }

    def export(self, onnx_path: str | None = None):
        """Runs export.

        @type onnx_path: str | None
        @param onnx_path: Path to .onnx model. If not specified, model will be saved
            to export directory with name specified in config file.

        @raises RuntimeError: If `onnxsim` fails to simplify the model.
        """
        onnx_path = onnx_path or self.export_path + ".onnx"
        self.output_names = self.lightning_module.export_onnx(
            onnx_path, **self.cfg.exporter.onnx.model_dump()
        )

        try:
            import onnxsim

            logger.info("Simplifying ONNX model...")
            model_onnx = onnx.load(onnx_path)
            onnx_model, check = onnxsim.simplify(model_onnx)
            if not check:
                raise RuntimeError("Onnx simplify failed.")
            onnx.save(onnx_model, onnx_path)
            logger.info(f"ONNX model saved to {onnx_path}")

        except ImportError:
            logger.error("Failed to import `onnxsim`")
            logger.warning(
                "`onnxsim` not installed. Skipping ONNX model simplification. "
                "Ensure `onnxsim` is installed in your environment."
            )

        files_to_upload = [self.local_path, onnx_path]

        if self.cfg.exporter.blobconverter.active:
            try:
                import blobconverter

                logger.info("Converting ONNX to .blob")

                optimizer_params = []
                if self.scale_values:
                    optimizer_params.append(f"--scale_values={self.scale_values}")
                if self.mean_values:
                    optimizer_params.append(f"--mean_values={self.mean_values}")
                if self.cfg.exporter.reverse_input_channels:
                    optimizer_params.append("--reverse_input_channels")

                blob_path = blobconverter.from_onnx(
                    model=onnx_path,
                    optimizer_params=optimizer_params,
                    data_type=self.cfg.exporter.data_type,
                    shaves=self.cfg.exporter.blobconverter.shaves,
                    use_cache=False,
                    output_dir=self.export_path,
                )
                files_to_upload.append(blob_path)
                logger.info(f".blob model saved to {blob_path}")

            except ImportError:
                logger.error("Failed to import `blobconverter`")
                logger.warning(
                    "`blobconverter` not installed. Skipping .blob model conversion. "
                    "Ensure `blobconverter` is installed in your environment."
                )

        if self.cfg.exporter.upload_url is not None:
            self._upload(files_to_upload)

    def _upload(self, files_to_upload: list[str]):
        """Uploads .pt, .onnx and current config.yaml to specified s3 bucket.

        @type files_to_upload: list[str]
        @param files_to_upload: List of files to upload.
        @raises ValueError: If upload url was not specified in config file.
        """

        if self.cfg.exporter.upload_url is None:
            raise ValueError("Upload url must be specified in config file.")

        fs = LuxonisFileSystem(self.cfg.exporter.upload_url, allow_local=False)
        logger.info(f"Started upload to {fs.full_path}...")

        for file in files_to_upload:
            suffix = Path(file).suffix
            fs.put_file(
                local_path=file,
                remote_path=self.cfg.exporter.export_model_name + suffix,
            )

        with tempfile.TemporaryFile() as f:
            self.cfg.save_data(f.name)
            fs.put_file(local_path=f.name, remote_path="config.yaml")

        onnx_path = os.path.join(
            fs.full_path, f"{self.cfg.exporter.export_model_name}.onnx"
        )
        modelconverter_config = self._get_modelconverter_config(onnx_path)

        with tempfile.TemporaryFile() as f:
            yaml.dump(modelconverter_config, f, default_flow_style=False)
            fs.put_file(local_path=f.name, remote_path="config_export.yaml")

        logger.info("Files upload finished")
