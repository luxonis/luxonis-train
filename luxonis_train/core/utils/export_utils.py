import os
from collections.abc import Generator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Literal

from loguru import logger
from luxonis_ml.typing import PathType, check_type

import luxonis_train as lxt
from luxonis_train.config import ExportConfig
from luxonis_train.config.config import HubAIExportConfig, PreprocessingConfig


@contextmanager
def replace_weights(
    module: "lxt.LuxonisLightningModule", weights: PathType | None = None
) -> Generator:
    old_weights = None
    if weights is not None:
        old_weights = module.state_dict()
        module.load_checkpoint(str(weights))

    yield

    if old_weights is not None:
        module.load_state_dict(old_weights)
        del old_weights


def try_onnx_simplify(onnx_path: PathType) -> None:
    import onnx

    try:
        import onnxsim

    except ImportError:
        logger.error("Failed to import `onnxsim`")
        logger.warning(
            "`onnxsim` not installed. Skipping ONNX model simplification. "
            "Ensure `onnxsim` is installed in your environment."
        )
        return

    logger.info("Simplifying ONNX model...")
    model_onnx = onnx.load(onnx_path)
    onnx_model, check = onnxsim.simplify(model_onnx)
    if not check:  # pragma: no cover
        logger.error(
            "Failed to simplify ONNX model. Proceeding without simplification."
        )
        return
    onnx.save(onnx_model, onnx_path)
    logger.info(f"ONNX model saved to {onnx_path}")


def get_preprocessing(
    cfg: PreprocessingConfig, log_label: str | None = None
) -> tuple[
    list[float] | None, list[float] | None, Literal["RGB", "BGR", "GRAY"]
]:
    def _get_norm_param(key: Literal["mean", "std"]) -> list[float] | None:
        params = cfg.normalize.params
        if key not in params:  # pragma: no cover
            if log_label is not None:
                logger.warning(
                    f"{log_label} requires the '{key}' "
                    "parameter to be present in "
                    "`trainer.preprocessing.normalize.params`. "
                    f"'{key}' normalization will not be applied."
                )
            return None
        param = params[key]
        if not check_type(param, list[float | int]):  # pragma: no cover
            if log_label is not None:
                logger.warning(
                    f"{log_label} requires the '{key}' parameter "
                    "of `trainer.preprocessing.normalize.params` "
                    f"to be a list of numbers. Got: {param}. "
                    f"'{key}' normalization will not be applied."
                )
            return None
        return [round(x * 255.0, 5) for x in param]

    return _get_norm_param("mean"), _get_norm_param("std"), cfg.color_space


def blobconverter_export(
    cfg: ExportConfig,
    scale_values: list[float] | None,
    mean_values: list[float] | None,
    reverse_channels: bool,
    export_path: PathType,
    onnx_path: PathType,
) -> Path:
    import blobconverter

    logger.info("Converting ONNX to .blob")

    optimizer_params: list[str] = []
    if scale_values:
        optimizer_params.append(f"--scale_values={scale_values}")
    if mean_values:
        optimizer_params.append(f"--mean_values={mean_values}")
    if reverse_channels:
        optimizer_params.append("--reverse_input_channels")

    blob_path = blobconverter.from_onnx(
        model=str(onnx_path),
        optimizer_params=optimizer_params,
        data_type=cfg.data_type.upper(),
        shaves=cfg.blobconverter.shaves,
        version=cfg.blobconverter.version,
        use_cache=False,
        output_dir=str(export_path),
    )
    logger.info(f".blob model saved to {blob_path}")
    return Path(blob_path)


def hubai_export(
    cfg: HubAIExportConfig,
    target_precision: str,
    archive_path: PathType,
    export_path: PathType,
) -> Path:
    """Convert an ONNX NNArchive to a platform-specific NNArchive using
    HubAI SDK.

    @type cfg: HubAIExportConfig
    @param cfg: HubAI export configuration containing platform and
        params.
    @type target_precision: str
    @param target_precision: Target precision (int8, fp16, fp32).
    @type archive_path: PathType
    @param archive_path: Path to the ONNX NNArchive to convert.
    @type export_path: PathType
    @param export_path: Directory where the converted archive will be
        saved.
    @rtype: Path
    @return: Path to the converted platform-specific NNArchive.
    """
    import uuid

    from hubai_sdk import HubAIClient

    hubai_token = os.environ.get("HUBAI_API_KEY")
    if not hubai_token:
        raise ValueError(
            "HUBAI_API_KEY environment variable is not set. "
            "Please set it to use HubAI SDK for model conversion. "
        )

    logger.info(  # defaults to rvc2 when platform is None
        f"Converting NNArchive to {(cfg.platform or 'rvc2').upper()} format using HubAI SDK"
    )

    precision_map = {
        "int8": "INT8",
        "fp16": "FP16",
        "fp32": "FP32",
    }
    precision = precision_map.get(target_precision.lower(), "FP16")

    client = HubAIClient(api_key=hubai_token)

    archive_path = Path(archive_path)

    # Generate a unique name to avoid conflicts with existing models on HubAI
    unique_suffix = uuid.uuid4().hex[:8]
    unique_name = f"luxonis-train-{unique_suffix}"

    base_kwargs: dict = {
        "path": str(archive_path),
        "name": unique_name,
        "target_precision": precision,
    }

    if cfg.params:
        base_kwargs.update(cfg.params)

    try:
        if cfg.platform == "hailo":
            if "quantization_data" not in base_kwargs:
                base_kwargs["quantization_data"] = "RANDOM"
            response = client.convert.Hailo(**base_kwargs)
        elif cfg.platform == "rvc3":
            response = client.convert.RVC3(**base_kwargs)
        elif cfg.platform == "rvc4":
            response = client.convert.RVC4(**base_kwargs)
        else:
            response = client.convert.RVC2(**base_kwargs)

        downloaded_path = Path(response.downloaded_path)

        export_path = Path(export_path)
        output_path = export_path / downloaded_path.name

        import shutil

        downloaded_parent = downloaded_path.parent

        shutil.move(downloaded_path, output_path)

        if downloaded_parent.exists() and downloaded_parent != Path.cwd():
            with suppress(OSError):
                downloaded_parent.rmdir()

        logger.info(f"HubAI converted archive saved to {output_path}")
        return output_path
    finally:
        # delete the temporary model created on HubAI
        try:
            client.models.delete_model(unique_name)
            logger.debug(f"Cleaned up temporary HubAI model: {unique_name}")
        except Exception as e:
            logger.warning(
                f"Failed to cleanup HubAI model '{unique_name}': {e}"
            )


def make_initializers_unique(onnx_path: PathType) -> None:
    """Each initializer that is used by multiple nodes gets duplicated
    so each node has its own copy.

    @type onnx_path: PathType
    @param onnx_path: Path to the ONNX model file to modify.
    """
    import copy
    from collections import defaultdict

    import onnx

    onnx_path = str(onnx_path)
    model = onnx.load(onnx_path)
    graph = model.graph

    initializer_info = {}
    for initializer in graph.initializer:
        initializer_info[initializer.name] = {
            "data": copy.deepcopy(initializer),
            "usage_count": 0,
        }

    if not initializer_info:
        logger.warning("No initializers found in the model")
        return

    for node in graph.node:
        for input_name in node.input:
            if input_name in initializer_info:
                initializer_info[input_name]["usage_count"] += 1

    name_mapping = defaultdict(list)
    new_initializers = []
    duplicated_count = 0

    for original_name, info in initializer_info.items():
        usage_count = info["usage_count"]

        if usage_count <= 1:
            name_mapping[original_name].append(original_name)
            new_initializers.append(info["data"])
        else:
            duplicated_count += 1
            for i in range(usage_count):
                new_name = f"{original_name}_unique_{i}"
                name_mapping[original_name].append(new_name)

                new_initializer = copy.deepcopy(info["data"])
                new_initializer.name = new_name
                new_initializers.append(new_initializer)

    del graph.initializer[:]
    graph.initializer.extend(new_initializers)

    usage_counters = dict.fromkeys(name_mapping, 0)

    for node in graph.node:
        new_inputs = []
        for input_name in node.input:
            if input_name in name_mapping:
                counter = usage_counters[input_name]
                new_name = name_mapping[input_name][counter]
                usage_counters[input_name] += 1
                new_inputs.append(new_name)
            else:
                new_inputs.append(input_name)

        del node.input[:]
        node.input.extend(new_inputs)

    onnx.save(model, onnx_path)

    try:
        onnx.checker.check_model(onnx_path)
    except Exception as e:
        logger.warning(
            f"ONNX checker failed after making initializers unique: {e}. "
            "If you encounter issues, try exporting with unique_onnx_initializers=False."
        )

    logger.info(
        f"Processed {len(initializer_info)} initializers: "
        f"{duplicated_count} shared initializers were duplicated"
    )
