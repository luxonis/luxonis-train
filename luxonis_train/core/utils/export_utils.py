from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from loguru import logger

import luxonis_train
from luxonis_train.config import Config, ExportConfig


@contextmanager
def replace_weights(
    module: "luxonis_train.models.LuxonisLightningModule",
    weights: str | Path | None = None,
) -> Generator:
    old_weights = None
    if weights is not None:
        old_weights = module.state_dict()
        module.load_checkpoint(str(weights))

    yield

    if old_weights is not None:
        try:
            module.load_state_dict(old_weights)
        except RuntimeError:
            logger.error(
                "Failed to strictly load old weights. The model likely underwent re-parametrization, "
                "which is a destructive operation. Loading old weights with strict=False."
            )
            module.load_state_dict(old_weights, strict=False)
        del old_weights


def try_onnx_simplify(onnx_path: str) -> None:
    import onnx

    try:
        import onnxsim

        logger.info("Simplifying ONNX model...")
        model_onnx = onnx.load(onnx_path)
        onnx_model, check = onnxsim.simplify(model_onnx)
        if not check:
            raise RuntimeError("ONNX simplify failed.")  # pragma: no cover
        onnx.save(onnx_model, onnx_path)
        logger.info(f"ONNX model saved to {onnx_path}")

    except ImportError:
        logger.error("Failed to import `onnxsim`")
        logger.warning(
            "`onnxsim` not installed. Skipping ONNX model simplification. "
            "Ensure `onnxsim` is installed in your environment."
        )
    except RuntimeError:  # pragma: no cover
        logger.error(
            "Failed to simplify ONNX model. Proceeding without simplification."
        )


def get_preprocessing(
    cfg: Config,
) -> tuple[list[float] | None, list[float] | None, bool]:
    normalize_params = cfg.trainer.preprocessing.normalize.params
    if cfg.exporter.scale_values is not None:
        scale_values = cfg.exporter.scale_values
    else:
        scale_values = normalize_params.get("std", None)
        if scale_values:
            scale_values = (
                [round(i * 255, 5) for i in scale_values]
                if isinstance(scale_values, list)
                else round(scale_values * 255, 5)
            )

    if cfg.exporter.mean_values is not None:
        mean_values = cfg.exporter.mean_values
    else:
        mean_values = normalize_params.get("mean", None)
        if mean_values:
            mean_values = (
                [round(i * 255, 5) for i in mean_values]
                if isinstance(mean_values, list)
                else round(mean_values * 255, 5)
            )
    reverse_channels = cfg.exporter.reverse_input_channels

    return scale_values, mean_values, reverse_channels


def blobconverter_export(
    cfg: ExportConfig,
    scale_values: list[float] | None,
    mean_values: list[float] | None,
    reverse_channels: bool,
    export_path: str,
    onnx_path: str,
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
        model=onnx_path,
        optimizer_params=optimizer_params,
        data_type=cfg.data_type.upper(),
        shaves=cfg.blobconverter.shaves,
        version=cfg.blobconverter.version,
        use_cache=False,
        output_dir=export_path,
    )
    logger.info(f".blob model saved to {blob_path}")
    return Path(blob_path)
