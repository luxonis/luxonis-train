from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

from loguru import logger
from luxonis_ml.typing import PathType, check_type

import luxonis_train as lxt
from luxonis_train.config import ExportConfig
from luxonis_train.config.config import PreprocessingConfig


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
