import math
from copy import deepcopy
from importlib.util import find_spec
from pathlib import Path
from typing import Any, cast

import torch
from aimet_torch import QuantizationSimModel
from aimet_torch.adaround.adaround_weight import Adaround, AdaroundParameters
from aimet_torch.batch_norm_fold import fold_all_batch_norms
from aimet_torch.bn_reestimation import reestimate_bn_stats
from aimet_torch.common.defs import QuantizationDataType, QuantScheme
from aimet_torch.common.quantsim_config.utils import (
    get_path_for_per_channel_config,
)
from aimet_torch.cross_layer_equalization import equalize_model
from aimet_torch.seq_mse import apply_seq_mse
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
    high_precision_patterns: list[str] | None = None,
    high_precision_bw: int = 16,
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

    # Raise quantization-sensitive branches to higher precision *before*
    # encodings are computed (seq_mse / compute_encodings below).
    if high_precision_patterns:
        _raise_module_precision(
            sim, high_precision_patterns, high_precision_bw
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


def _raise_module_precision(
    sim: QuantizationSimModel, patterns: list[str], bitwidth: int
) -> None:
    """Set output/param quantizers of matching modules to ``bitwidth``.

    A module matches if any string in ``patterns`` is a substring of its
    fully-qualified name in ``sim.model``. This is used to keep
    quantization-sensitive branches (e.g. direct keypoint coordinate
    regression, whose decode amplifies activation error by ``2 *
    stride``) at higher precision than the rest of the network.

    Must be called *before* encodings are computed so the chosen
    bitwidth is reflected in the calibrated min/max ranges.
    """
    matched = 0
    for name, module in sim.model.named_modules():
        if not any(p in name for p in patterns):
            continue
        quantizers = []
        for attr in ("output_quantizers", "param_quantizers"):
            container = getattr(module, attr, None)
            if container is None:
                continue
            values = (
                container.values()
                if isinstance(container, dict)
                else container
            )
            quantizers.extend(q for q in values if q is not None)
        if not quantizers:
            continue
        for quantizer in quantizers:
            if hasattr(quantizer, "bitwidth"):
                quantizer.bitwidth = bitwidth
        matched += 1
        logger.info(
            f"Raised quantizers of '{name}' to {bitwidth}-bit precision"
        )

    if matched == 0:
        logger.warning(
            "high_precision_patterns matched no quantized modules "
            f"(patterns={patterns}); precision override had no effect"
        )


def _clear_inference_tensors(model: nn.Module) -> None:
    """Clone inference-mode tensors in all modules to normal tensors.

    Lightning's trainer.test() runs under torch.inference_mode(), so any
    tensor lazily created inside a module's forward during testing (e.g.
    stride_tensor in loss modules) becomes an inference tensor.  These
    cannot be saved for backward, which breaks QAT.  Cloning them
    outside inference_mode produces normal, autograd-compatible tensors.

    Loss/metric/visualizer modules are stored as plain Python dicts in
    NodeWrapper (not nn.ModuleDict) so model.modules() does not reach
    them. This function therefore also recurses into nn.Module instances
    found inside plain dict attributes.
    """
    visited: set[int] = set()

    def _visit(module: nn.Module) -> None:
        if id(module) in visited:
            return
        visited.add(id(module))

        for name, buf in list(module._buffers.items()):
            if buf is not None and buf.is_inference():
                module._buffers[name] = buf.clone()
        for name, val in list(module.__dict__.items()):
            if (
                isinstance(val, Tensor)
                and name not in module._buffers
                and name not in module._parameters
                and val.is_inference()
            ):
                module.__dict__[name] = val.clone()

        for child in module.children():
            _visit(child)

        # Recurse into nn.Module instances stored in plain dict attributes
        # (e.g. NodeWrapper.losses / .metrics / .visualizers).  These are
        # invisible to model.modules() but still hold inference tensors.
        for attr_val in module.__dict__.values():
            if isinstance(attr_val, dict) and not isinstance(
                attr_val, nn.ModuleDict
            ):
                for item in attr_val.values():
                    if isinstance(item, nn.Module):
                        _visit(item)

    _visit(model)


def quantization_aware_training(
    sim: QuantizationSimModel,
    dummy_inputs: Tensor,
    train_loader: DataLoader,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    epochs: int,
    fold_batch_norms: bool = False,
    batch_norm_reestimation: bool = False,
    *,
    val_loader: DataLoader | None = None,
    pl_trainer: Any = None,
    pre_quant_test: dict[str, float] | None = None,
    ptq_test: dict[str, float] | None = None,
    save_dir: Path | None = None,
    main_metric_key: str | None = None,
    validation_interval: int = 0,
) -> LuxonisLightningModule:

    model = cast(LuxonisLightningModule, sim.model)

    # Tensors lazily created during trainer.test() (which runs under
    # torch.inference_mode()) cannot participate in autograd.  Clone them now,
    # before training begins, so backward passes work correctly.
    _clear_inference_tensors(model)

    model.train()
    if CUDAAccelerator.is_available():
        model.cuda()
    model.automatic_optimization = False

    best_metric: float | None = None
    best_state_dict: dict[str, Any] | None = None

    for epoch in range(epochs):
        for imgs, labels in track(
            train_loader,
            description=f"QAT epoch {epoch + 1}/{epochs}",
            total=len(train_loader),
        ):
            optimizer.zero_grad()
            loss = model.training_step((imgs, labels))
            loss.backward()
            optimizer.step()
        scheduler.step()

        if (
            validation_interval > 0
            and val_loader is not None
            and pl_trainer is not None
            and (epoch + 1) % validation_interval == 0
        ):
            model.eval()
            qat_test = pl_trainer.test(model, val_loader)[0]
            _clear_inference_tensors(model)
            model.train()

            if pre_quant_test is not None and ptq_test is not None:
                table = [
                    (
                        key.replace("test/metric/", "").replace(
                            "test/loss/", ""
                        ),
                        value,
                        ptq_test[key],
                        qat_test[key],
                    )
                    for key, value in pre_quant_test.items()
                    if key in qat_test and key in ptq_test
                ]
                model.progress_bar.print_table(
                    f"Quantization results (epoch {epoch + 1}/{epochs})",
                    table,
                    ["Name", "Pre-Quant", "PTQ", "QAT"],
                )

            if main_metric_key is not None and save_dir is not None:
                current_metric = qat_test.get(main_metric_key)
                if current_metric is not None and (
                    best_metric is None or current_metric > best_metric
                ):
                    best_metric = current_metric
                    best_state_dict = {
                        k: v.detach().clone()
                        if isinstance(v, Tensor)
                        else deepcopy(v)
                        for k, v in model.state_dict().items()
                    }
                    ckpt_path = save_dir / "best_qat.pt"
                    torch.save(
                        {
                            "state_dict": {
                                k: v.cpu() if isinstance(v, Tensor) else v
                                for k, v in best_state_dict.items()
                            }
                        },
                        ckpt_path,
                    )
                    logger.info(
                        f"New best QAT model (metric={current_metric:.6f}) "
                        f"saved to {ckpt_path}"
                    )

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        logger.info("Restored best QAT model weights for final export")

    if batch_norm_reestimation:
        logger.info("Reestimating batch norm statistics")

        reestimate_bn_stats(
            model, train_loader, forward_fn=_patched_forward_pass
        )

        # fold_all_batch_norms cannot be called on the quantized sim model:
        # AIMET's graph tracer (ConnectedGraph) uses torch.jit.trace, which
        # breaks on quantization-wrapped modules whose input count differs from
        # the original (e.g. DFL gets 5 inputs instead of 1 once wrapped).
        # BNs with updated stats are exported correctly via sim.onnx.export(),
        # and onnxsim folds them at the ONNX level.

    model.automatic_optimization = True
    return model


def _patched_forward_pass(
    model: nn.Module, inputs: LuxonisLoaderTorchOutput
) -> Any:
    return model(inputs[0])
