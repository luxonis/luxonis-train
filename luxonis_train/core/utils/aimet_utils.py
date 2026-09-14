"""AIMET post-training quantization and quantization-aware training.

`LuxonisModel.quantize` calls these helpers. AIMET is an optional
dependency, installed with the ``aimet`` extra of ``luxonis-train``. The
module imports ``aimet_torch`` only inside the functions, so the module
itself imports without AIMET.

"""

import math
from collections.abc import Callable, Sized
from importlib.util import find_spec
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch.utils.data as torch_data
from lightning.pytorch.accelerators import CUDAAccelerator
from loguru import logger
from rich.progress import track
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from luxonis_train.lightning import LuxonisLightningModule
from luxonis_train.loaders.base_loader import LuxonisLoaderTorchOutput

if TYPE_CHECKING:
    from aimet_torch import (  # pyright: ignore[reportMissingImports]
        QuantizationSimModel,
    )
    from aimet_torch.common.defs import (  # pyright: ignore[reportMissingImports]
        QuantizationDataType,
        QuantScheme,
    )


def check_aimet_available() -> None:
    """Raise an error when the ``aimet_torch`` package is not installed.

    The function looks for the package with
    `importlib.util.find_spec`. It does not import the package.

    Raises:
        ImportError: When ``aimet_torch`` is not installed. The message
            tells the user to install ``luxonis-train[aimet]``.

    """
    if not find_spec("aimet_torch"):
        raise ImportError(
            "AIMET library is not installed. Please install "
            "`luxonis-train` with the `aimet` extra enabled "
            "(pip install luxonis-train[aimet])"
        )


def get_ptq_calibration_loader(
    val_dataset: torch_data.Dataset[LuxonisLoaderTorchOutput],
    collate_fn: Callable[[list[LuxonisLoaderTorchOutput]], Any],
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    max_calibration_images: int | None,
) -> DataLoader:
    """Build the loader that calibrates the post-training quantization.

    `LuxonisModel.quantize` passes the validation dataset itself, not
    `LuxonisModel.val_loader`. So ``trainer.n_validation_batches`` does
    not limit the calibration.

    With ``max_calibration_images``, the loader reads only the first
    ``max_calibration_images`` samples of ``val_dataset``. The function
    logs an info message when this leaves out samples. When
    ``val_dataset`` has fewer samples than requested, the function logs
    a warning and uses all samples. The loader keeps the order of the
    dataset and keeps the last incomplete batch.

    Args:
        val_dataset (``torch_data.Dataset[LuxonisLoaderTorchOutput]``):
            The dataset of the validation view. It must support ``len``
            when ``max_calibration_images`` is not ``None``.
        collate_fn (``Callable[[list[LuxonisLoaderTorchOutput]], Any]``):
            The function that merges a list of samples into a batch,
            such as `BaseLoaderTorch.collate_fn`.
        batch_size (int): The number of samples in a batch.
        num_workers (int): The number of worker processes of the
            loader. ``0`` loads the samples in the main process.
        pin_memory (bool): Copy the tensors of each batch into pinned
            memory before the loader returns them.
        max_calibration_images (int | None): The maximum number of
            samples to read. ``None`` reads all samples.

    Returns:
        torch.utils.data.DataLoader: The calibration loader.

    Example:
        The example turns the logger off, so the info message does not
        show:

        >>> import torch
        >>> from loguru import logger
        >>> from torch.utils.data import TensorDataset, default_collate
        >>> logger.disable("luxonis_train")
        >>> loader = get_ptq_calibration_loader(
        ...     TensorDataset(torch.arange(10)),
        ...     collate_fn=default_collate,
        ...     batch_size=4,
        ...     num_workers=0,
        ...     pin_memory=False,
        ...     max_calibration_images=6,
        ... )
        >>> logger.enable("luxonis_train")
        >>> [batch.tolist() for (batch,) in loader]
        [[0, 1, 2, 3], [4, 5]]

    """
    loader: torch_data.Dataset[LuxonisLoaderTorchOutput] = val_dataset
    if max_calibration_images is not None:
        dataset_size = len(cast(Sized, loader))
        subset_size = min(max_calibration_images, dataset_size)
        if max_calibration_images > dataset_size:
            logger.warning(
                "PTQ calibration requested "
                f"{max_calibration_images} images, but the validation dataset "
                f"only has {dataset_size}. Using the available {dataset_size}."
            )
        elif subset_size < dataset_size:
            logger.info(
                "Limiting PTQ calibration to the first "
                f"{subset_size} / {dataset_size} validation samples because "
                f"`exporter.aimet.max_calibration_images={max_calibration_images}`."
            )
        loader = torch_data.Subset(loader, range(subset_size))

    return torch_data.DataLoader(
        loader,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
        shuffle=False,
        drop_last=False,
        pin_memory=pin_memory,
    )


def post_training_quantization(
    model: LuxonisLightningModule,
    dummy_inputs: Tensor,
    val_loader: DataLoader,
    save_dir: Path,
    quant_scheme: "QuantScheme | None" = None,
    default_output_bw: int = 8,
    default_param_bw: int = 8,
    default_data_type: "QuantizationDataType | None" = None,
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
) -> "QuantizationSimModel":
    r"""Quantize a module with AIMET after training.

    The function runs these steps:

    - It moves ``model`` and ``dummy_inputs`` to the GPU when CUDA is
      available, and puts ``model`` in eval mode.
    - With ``fold_batch_norms`` and without ``batch_norm_reestimation``,
      it folds the batch norms of ``model`` into the preceding layers.
    - With ``cross_layer_equalization``, it equalizes the weight ranges
      of consecutive layers of ``model``.
    - With ``adaround``, it learns the rounding of the weights on at
      most :math:`\lceil 2000 / B \rceil` batches of ``val_loader``,
      where :math:`B` is the batch size. AdaRound writes its files with
      the prefix ``adaround`` to ``save_dir``. The next steps use the
      module that AdaRound returns instead of ``model``.
    - It builds a ``QuantizationSimModel`` around the module with
      ``in_place=True``.
    - With ``sequential_mse``, it applies sequential MSE on
      ``val_loader`` with 20 candidates.
    - With ``adaround``, it loads ``adaround.encodings`` from
      ``save_dir`` and freezes these parameter encodings.
    - It computes the encodings with a forward pass of the inputs of
      every batch of ``val_loader``, and shows a progress bar.

    The steps log an info message for the batch norm folding, the
    cross-layer equalization, and the sequential MSE.

    Args:
        model (LuxonisLightningModule): The module to quantize.
        dummy_inputs (``Tensor``): An input batch for the graph traces,
            such as a random tensor of shape ``[1, C, H, W]``.
        val_loader (torch.utils.data.DataLoader): The calibration
            loader, such as the result of `get_ptq_calibration_loader`.
            Each batch is a pair of the inputs and the labels.
        save_dir (``Path``): The directory for the AdaRound files.
        quant_scheme (``QuantScheme | None``): The AIMET quantization
            scheme. ``None`` selects ``QuantScheme.min_max``.
        default_output_bw (int): The bit width of the activations.
        default_param_bw (int): The bit width of the parameters.
        default_data_type (``QuantizationDataType | None``): The data
            type of a quantized value. ``None`` selects
            ``QuantizationDataType.int``.
        config_file (str | None): The path of an AIMET config JSON file.
            ``None`` with ``batch_norm_reestimation`` selects the
            per-channel config of AIMET.
        adaround (bool): Apply AdaRound.
        adaround_iterations (int | None): The number of AdaRound
            iterations, passed to ``AdaroundParameters``.
        adaround_reg_param (float): The AdaRound regularization
            parameter.
        adaround_beta_range (tuple[int, int]): The start and the end of
            the AdaRound beta annealing.
        adaround_warm_start (float): The share of the AdaRound
            iterations during which the rounding loss has no effect.
        fold_batch_norms (bool): Fold the batch norms before
            quantization. It has no effect with
            ``batch_norm_reestimation``, because
            `quantization_aware_training` then folds them.
        cross_layer_equalization (bool): Apply cross-layer
            equalization.
        batch_norm_reestimation (bool): Whether
            `quantization_aware_training` re-estimates the batch norms.
            With ``True``, the function skips the batch norm folding.
            It also selects the per-channel config of AIMET when
            ``config_file`` is ``None``.
        sequential_mse (bool): Apply sequential MSE.

    Returns:
        ``QuantizationSimModel``: The simulation, with the computed
        encodings. Its ``model`` is the quantized module.

    Raises:
        ImportError: When ``aimet_torch`` is not installed.
        AssertionError: When ``val_loader`` has no batch.

    """
    check_aimet_available()

    from aimet_torch import (  # pyright: ignore[reportMissingImports]
        QuantizationSimModel,
    )
    from aimet_torch.adaround.adaround_weight import (  # pyright: ignore[reportMissingImports]
        Adaround,
        AdaroundParameters,
    )
    from aimet_torch.batch_norm_fold import (  # pyright: ignore[reportMissingImports]
        fold_all_batch_norms,
    )
    from aimet_torch.common.defs import (  # pyright: ignore[reportMissingImports]
        QuantizationDataType,
        QuantScheme,
    )
    from aimet_torch.common.quantsim_config.utils import (  # pyright: ignore[reportMissingImports]
        get_path_for_per_channel_config,
    )
    from aimet_torch.cross_layer_equalization import (  # pyright: ignore[reportMissingImports]
        equalize_model,
    )
    from aimet_torch.seq_mse import (  # pyright: ignore[reportMissingImports]
        apply_seq_mse,
    )

    quant_scheme = (
        QuantScheme.min_max if quant_scheme is None else quant_scheme
    )
    default_data_type = (
        QuantizationDataType.int
        if default_data_type is None
        else default_data_type
    )

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
    sim: "QuantizationSimModel",
    dummy_inputs: Tensor,
    train_loader: DataLoader,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    epochs: int,
    fold_batch_norms: bool = False,
    batch_norm_reestimation: bool = False,
) -> LuxonisLightningModule:
    """Train the quantized module of an AIMET simulation.

    The function trains ``sim.model`` in place, with the quantizers in
    the forward pass. It puts the module in training mode, and moves it
    to the GPU when CUDA is available. It turns off the automatic
    optimization of the module for the loop, and restores the previous
    value at the end, also after an error.

    For each epoch, the function runs every batch of ``train_loader``
    with a progress bar. For each batch, it computes the loss with
    `LuxonisLightningModule.compute_training_loss`, runs the backward
    pass, and steps ``optimizer``. It steps ``scheduler`` once at the
    end of each epoch.

    With ``batch_norm_reestimation``, the function then re-estimates the
    batch norm statistics on ``train_loader``. With
    ``fold_batch_norms`` too, it folds the batch norms into the
    preceding layers. When AIMET cannot trace the graph of the quantized
    module, the function logs a warning and skips the folding.

    Args:
        sim (``QuantizationSimModel``): The simulation from
            `post_training_quantization`.
        dummy_inputs (``Tensor``): An input batch for the graph trace of
            the batch norm folding.
        train_loader (torch.utils.data.DataLoader): The training loader.
            Each batch is a pair of the inputs and the labels.
        optimizer (``Optimizer``): The optimizer of the parameters of
            ``sim.model``.
        scheduler (``LRScheduler``): The learning rate scheduler of
            ``optimizer``.
        epochs (int): The number of passes over ``train_loader``.
        fold_batch_norms (bool): Fold the batch norms after the
            re-estimation. It has no effect without
            ``batch_norm_reestimation``.
        batch_norm_reestimation (bool): Re-estimate the batch norm
            statistics after the training.

    Returns:
        LuxonisLightningModule: ``sim.model`` after the training. The
        function does not put it back in eval mode.

    Raises:
        ImportError: When ``aimet_torch`` is not installed.
        AssertionError: When ``train_loader`` has no batch.

    """
    check_aimet_available()

    from aimet_torch.batch_norm_fold import (  # pyright: ignore[reportMissingImports]
        fold_all_batch_norms,
    )
    from aimet_torch.bn_reestimation import (  # pyright: ignore[reportMissingImports]
        reestimate_bn_stats,
    )

    model = cast(LuxonisLightningModule, sim.model)

    model.train()
    if CUDAAccelerator.is_available():
        model.cuda()
    previous_automatic_optimization = model.automatic_optimization
    model.automatic_optimization = False

    try:
        assert len(train_loader) > 0, (
            "Training loader must have at least one batch"
        )

        for epoch in range(epochs):
            for imgs, labels in track(
                train_loader,
                description=(
                    "Running Quantization-Aware Training "
                    f"(epoch {epoch + 1}/{epochs})"
                ),
                total=len(train_loader),
            ):
                optimizer.zero_grad()
                loss = model.compute_training_loss((imgs, labels))
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
                try:
                    fold_all_batch_norms(
                        model,
                        input_shapes=dummy_inputs.shape,
                        dummy_input=dummy_inputs,
                    )
                except Exception as e:  # pragma: no cover
                    if not _is_aimet_graph_trace_error(e):  # pragma: no cover
                        raise
                    logger.warning(
                        "Skipping post-QAT batch norm folding because AIMET "
                        "failed to trace the quantized model graph. "
                        f"Error: {e}"
                    )
    finally:
        model.automatic_optimization = previous_automatic_optimization
    return model


def _is_aimet_graph_trace_error(exc: Exception) -> bool:  # pragma: no cover
    return exc.__class__.__name__ == "_UnsafeGraphError" or (
        "Failed to trace computation graph" in str(exc)
    )


def _patched_forward_pass(
    model: nn.Module, inputs: LuxonisLoaderTorchOutput
) -> Any:
    return model(inputs[0])
