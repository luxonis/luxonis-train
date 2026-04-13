import math
from importlib.util import find_spec
from pathlib import Path
from typing import Any, cast

from aimet_torch import QuantizationSimModel
from aimet_torch.adaround.adaround_weight import Adaround, AdaroundParameters
from aimet_torch.bn_reestimation import reestimate_bn_stats
from aimet_torch.common.defs import QuantizationDataType, QuantScheme
from aimet_torch.common.quantsim_config.utils import (
    get_path_for_per_channel_config,
)
from aimet_torch.cross_layer_equalization import equalize_model
from aimet_torch.seq_mse import apply_seq_mse
from aimet_torch.v1.batch_norm_fold import fold_all_batch_norms
from lightning.pytorch.accelerators import CUDAAccelerator
from loguru import logger
from rich.progress import track
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from luxonis_train.lightning import LuxonisLightningModule
from luxonis_train.loaders.base_loader import LuxonisLoaderTorchOutput


def check_aimet_available() -> None:
    if not find_spec("aimet_torch"):
        raise ImportError(
            "AIMET library is not installed. Please install "
            "`luxonis-train` with the `aimet` extra enabled "
            "(pip install luxonis-train[aimet] --extra-index-url https://download.pytorch.org/whl/cu126)"
        )


def post_training_quantization(
    model: LuxonisLightningModule,
    dummy_inputs: Tensor,
    val_loader: DataLoader,
    save_dir: Path,
    quant_scheme: str | QuantScheme = QuantScheme.min_max,
    default_output_bw: int = 8,
    default_param_bw: int = 8,
    default_data_type: QuantizationDataType = QuantizationDataType.int,
    config_file: str | None = None,
    adaround: bool = False,
    adaround_iterations: int | None = None,
    adaround_reg_param: float = 0.01,
    adaround_beta_range: tuple[int, int] = (20, 2),
    adaround_warm_start: float = 0.2,
    fold_batch_norms: bool = False,
    cross_layer_equalization: bool = False,
    batch_norm_reestimation: bool = False,
    sequential_mse: bool = False,
) -> QuantizationSimModel:

    def pass_calibration_data(model: nn.Module) -> None:
        assert len(val_loader) > 0, (
            "Validation loader must have at least one batch"
        )
        for imgs, _ in track(
            val_loader,
            description="Computing quantization encodings",
            total=len(val_loader),
        ):
            model.forward(imgs)

    if CUDAAccelerator.is_available():
        dummy_inputs = dummy_inputs.cuda()
        model.cuda()

    model.eval()

    if fold_batch_norms and not batch_norm_reestimation:
        logger.info("Folding batch norms into preceding layers")
        fold_all_batch_norms(
            model, input_shapes=dummy_inputs.shape, dummy_input=dummy_inputs
        )
    if cross_layer_equalization:
        logger.info("Applying cross-layer equalization")
        equalize_model(
            model, input_shapes=dummy_inputs.shape, dummy_input=dummy_inputs
        )

    if adaround:
        ada_params = AdaroundParameters(
            data_loader=val_loader,
            num_batches=min(
                len(val_loader),
                math.ceil(2000 / val_loader.batch_size),  # type: ignore
            ),
            default_num_iterations=adaround_iterations,  # type: ignore
            default_reg_param=adaround_reg_param,
            default_beta_range=adaround_beta_range,
            default_warm_start=adaround_warm_start,
        )
        model = cast(
            LuxonisLightningModule,
            Adaround.apply_adaround(
                model,
                dummy_inputs,
                ada_params,
                path=str(save_dir),
                filename_prefix="adaround",
            ),
        )

    if batch_norm_reestimation and config_file is None:
        config_file = get_path_for_per_channel_config()

    sim = QuantizationSimModel(
        model=model,
        dummy_input=dummy_inputs,
        quant_scheme=quant_scheme,
        default_output_bw=default_output_bw,
        default_param_bw=default_param_bw,
        config_file=config_file,
        default_data_type=default_data_type,
        in_place=True,
    )
    if sequential_mse:
        logger.info("Applying sequential MSE")

        apply_seq_mse(
            sim,
            data_loader=val_loader,
            num_candidates=20,
            forward_fn=_patched_forward_pass,
        )

    if adaround:
        sim.set_and_freeze_param_encodings(
            str(save_dir / "adaround.encodings")
        )

    sim.compute_encodings(pass_calibration_data)
    return sim


def quantization_aware_training(
    sim: QuantizationSimModel,
    dummy_inputs: Tensor,
    train_loader: DataLoader,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    epochs: int,
    fold_batch_norms: bool = False,
    batch_norm_reestimation: bool = False,
) -> LuxonisLightningModule:

    model = cast(LuxonisLightningModule, sim.model)

    model.train()
    if CUDAAccelerator.is_available():
        model.cuda()
    model.automatic_optimization = False

    for _ in track(
        range(epochs),
        description="Running Quantization-Aware Training",
        total=epochs,
    ):
        for imgs, labels in train_loader:
            optimizer.zero_grad()
            loss = model.training_step((imgs, labels))
            loss.backward()
            optimizer.step()
        scheduler.step()

    if batch_norm_reestimation:
        logger.info("Reestimating batch norm statistics")

        reestimate_bn_stats(
            model, train_loader, forward_fn=_patched_forward_pass
        )

        if fold_batch_norms:
            logger.info("Folding batch norms into preceding layers")
            fold_all_batch_norms(
                model,
                input_shapes=dummy_inputs.shape,
                dummy_input=dummy_inputs,
            )

    model.automatic_optimization = True
    return model


def _patched_forward_pass(
    model: nn.Module, inputs: LuxonisLoaderTorchOutput
) -> Any:
    return model(inputs[0])
