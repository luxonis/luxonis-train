"""Helpers of the ONNX export and of the conversions that follow it.

`LuxonisModel.export`, `LuxonisModel.archive`, `LuxonisModel.convert`,
and `LuxonisModel.quantize` call these helpers. The helpers do these
steps:

- simplify the ONNX graph and duplicate its shared initializers;
- rename the graph outputs;
- read the normalization of the config;
- convert the model with ``blobconverter`` or the HubAI SDK.

`replace_weights` also loads the weights for `LuxonisModel.test`,
`LuxonisModel.infer`, and `LuxonisModel.annotate`.

"""

import copy
import os
import shutil
from collections import defaultdict
from collections.abc import Generator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypedDict

from loguru import logger
from luxonis_ml.typing import PathType, check_type

import luxonis_train as lxt
from luxonis_train.config import ExportConfig
from luxonis_train.config.config import HubAIExportConfig, PreprocessingConfig

if TYPE_CHECKING:
    from hubai_sdk import HubAIClient
    from hubai_sdk.utils.sdk_models import ConvertResponse
    from onnx import GraphProto, TensorProto


@contextmanager
def replace_weights(
    module: "lxt.LuxonisLightningModule",
    weights: PathType | dict[str, Any] | None = None,
) -> Generator:
    """Load ``weights`` into ``module`` inside a ``with`` block.

    On entry, when ``weights`` is not ``None``, the manager keeps the
    result of ``module.state_dict()``. It then loads ``weights`` with
    `LuxonisLightningModule.load_checkpoint`, which also puts the module
    in evaluation mode. It sets the private flag
    ``_weights_explicitly_loaded`` on the module. While that flag is
    set, `EMACallback` does not swap the module to the EMA weights and
    does not swap it back. On exit, also after an error in the block,
    the manager clears the flag and loads the kept state dict into the
    module. It does not restore the training mode that the module had
    before the block.

    **The kept state dict is not a copy.** Its tensors share memory
    with the module, so the load of ``weights`` overwrites them too.
    After the block, the module still holds ``weights``.

    With ``weights`` of ``None``, the block runs with the module
    unchanged.

    Args:
        module (LuxonisLightningModule): The module that receives the
            weights.
        weights (``PathType | dict[str, Any] | None``): A path to a
            checkpoint file, or a loaded checkpoint with a
            ``state_dict`` key. ``None`` leaves the module unchanged.

    Yields:
        None: The manager yields no value. The block runs with
        ``weights`` loaded.

    """
    old_weights = None
    if weights is not None:
        old_weights = module.state_dict()
        module.load_checkpoint(weights)
        object.__setattr__(module, "_weights_explicitly_loaded", True)

    try:
        yield
    finally:
        if old_weights is not None:
            object.__setattr__(module, "_weights_explicitly_loaded", False)
            module.load_state_dict(old_weights)
            del old_weights


def try_onnx_simplify(onnx_path: PathType) -> None:
    """Simplify an ONNX model in place with ``onnxsim``, when available.

    The function loads the model, runs ``onnxsim.simplify``, and saves
    the result over ``onnx_path``. It logs an error and leaves the file
    unchanged in these cases:

    - ``onnxsim`` is not installed. The function also logs a warning.
    - The check of ``onnxsim`` reports that the simplified model is not
      valid.

    Args:
        onnx_path (``PathType``): The ONNX file to simplify.

    """
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


def rename_onnx_outputs(onnx_path: PathType, output_names: list[str]) -> None:
    """Rename the graph outputs of an ONNX model in place.

    The function pairs the outputs of the graph with ``output_names``
    in order. It renames each graph output, and the output of every
    node that produces it. It does not rename the inputs of the nodes
    that read such an output.

    When the file ``<onnx_path.name>.data`` exists next to the model,
    the function deletes it after the load. It then saves the model
    over ``onnx_path``, with the initializers in a new external data
    file of that name. The tensors of the node attributes stay in the
    model file. Otherwise the function saves the whole model over
    ``onnx_path``. It then checks the saved file with
    ``onnx.checker.check_model``. The check raises an error when the
    model is not valid.

    Args:
        onnx_path (``PathType``): The ONNX file to modify.
        output_names (list[str]): The new output names, one for each
            graph output, in graph order.

    Raises:
        ValueError: When the length of ``output_names`` differs from
            the number of graph outputs.

    """
    import onnx

    onnx_path = Path(onnx_path)
    model = onnx.load(str(onnx_path))

    if len(model.graph.output) != len(output_names):
        raise ValueError(
            "Number of requested output names does not match the number "
            f"of ONNX graph outputs: {len(output_names)} != "
            f"{len(model.graph.output)}."
        )

    old_to_new = {
        output.name: new_name
        for output, new_name in zip(
            model.graph.output, output_names, strict=True
        )
    }

    for node in model.graph.node:
        for i, name in enumerate(node.output):
            if name in old_to_new:
                node.output[i] = old_to_new[name]

    for output in model.graph.output:
        if output.name in old_to_new:
            output.name = old_to_new[output.name]

    external_data_path = onnx_path.with_name(f"{onnx_path.name}.data")
    if external_data_path.exists():
        external_data_path.unlink()
        onnx.save(
            model,
            str(onnx_path),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=external_data_path.name,
            size_threshold=0,
            convert_attribute=False,
        )
    else:
        onnx.save(model, str(onnx_path))

    onnx.checker.check_model(str(onnx_path))


def get_preprocessing(
    cfg: PreprocessingConfig, log_label: str | None = None
) -> tuple[
    list[float] | None, list[float] | None, Literal["RGB", "BGR", "GRAY"]
]:
    """Read the normalization values and the color space of a config.

    The mean and the standard deviation come from
    ``cfg.normalize.params``, multiplied by ``255`` and rounded to
    five decimals, so they apply to ``uint8`` pixel values.
    `LuxonisModel` puts these values into the ``modelconverter``
    config and the NN Archive, and passes them to ``blobconverter``.
    ``exporter.mean_values`` and ``exporter.scale_values`` replace
    them there when they are set.

    A value is ``None`` in these cases:

    - ``cfg.normalize.active`` is ``False``. Both values are then
      ``None``.
    - ``cfg.normalize.params`` has no ``"mean"`` or ``"std"`` key for
      the value.
    - The value under the key is not a list of numbers.

    In the last two cases, the function logs a warning when
    ``log_label`` is not ``None``. The warning names the caller and
    the value that stays unset.

    Args:
        cfg (PreprocessingConfig): The ``trainer.preprocessing``
            section of the config.
        log_label (str | None): The name of the caller in the warning,
            such as ``"Model export"``. ``None`` disables the warning.

    Returns:
        ``tuple[list[float] | None, list[float] | None, Literal["RGB", "BGR", "GRAY"]]``:
        The mean values, the standard deviation values, and
        ``cfg.color_space``.

    Examples:
        >>> from luxonis_train.config.config import PreprocessingConfig
        >>> get_preprocessing(PreprocessingConfig())
        ([123.675, 116.28, 103.53], [58.395, 57.12, 57.375], 'RGB')

        >>> get_preprocessing(PreprocessingConfig(normalize={"active": False}))
        (None, None, 'RGB')

        >>> cfg = PreprocessingConfig(normalize={"params": {"mean": [0.5]}})
        >>> get_preprocessing(cfg)
        ([127.5], None, 'RGB')

    """
    if not cfg.normalize.active:
        return None, None, cfg.color_space

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
    """Convert an ONNX model to a ``.blob`` file with ``blobconverter``.

    The function calls ``blobconverter.from_onnx`` with
    ``cfg.blobconverter.shaves`` and ``cfg.blobconverter.version``,
    and with the cache off. It passes ``scale_values``,
    ``mean_values``, and ``reverse_channels`` to the model optimizer
    as ``--scale_values=[...]``, ``--mean_values=[...]``, and
    ``--reverse_input_channels``. It leaves out a value that is
    ``None``, empty, or ``False``. ``blobconverter`` sends the model to
    its online service, so the conversion needs network access.

    ``cfg.quantization_mode`` selects the data type. The mode
    ``"FP16_STANDARD"`` gives ``FP16``, and ``"FP32_STANDARD"`` gives
    ``FP32``. Any other mode, such as the default ``"INT8_STANDARD"``,
    gives ``FP16`` and logs a warning.

    Args:
        cfg (ExportConfig): The ``exporter`` section of the config.
        scale_values (list[float] | None): The scale of the input
            normalization, per channel, in ``uint8`` pixel units. The
            standard deviation from `get_preprocessing` has this
            format.
        mean_values (list[float] | None): The mean of the input
            normalization, per channel, in ``uint8`` pixel units. The
            mean from `get_preprocessing` has this format.
        reverse_channels (bool): When ``True``, pass
            ``--reverse_input_channels``, which swaps the order of the
            input channels.
        export_path (``PathType``): The directory that receives the
            ``.blob`` file.
        onnx_path (``PathType``): The ONNX file to convert.

    Returns:
        ``Path``: The path of the ``.blob`` file.

    """
    import blobconverter

    logger.info("Converting ONNX to .blob")

    optimizer_params: list[str] = []
    if scale_values:
        optimizer_params.append(f"--scale_values={scale_values}")
    if mean_values:
        optimizer_params.append(f"--mean_values={mean_values}")
    if reverse_channels:
        optimizer_params.append("--reverse_input_channels")

    # Map quantization_mode to blobconverter data_type
    # blobconverter only supports FP16 and FP32.
    quantization_to_dtype = {
        "FP16_STANDARD": "FP16",
        "FP32_STANDARD": "FP32",
    }
    data_type = quantization_to_dtype.get(cfg.quantization_mode, "FP16")
    if cfg.quantization_mode not in quantization_to_dtype:
        logger.warning(
            f"blobconverter does not support '{cfg.quantization_mode}' quantization. "
            f"Falling back to 'FP16'."
        )

    blob_path = blobconverter.from_onnx(
        model=str(onnx_path),
        optimizer_params=optimizer_params,
        data_type=data_type,
        shaves=cfg.blobconverter.shaves,
        version=cfg.blobconverter.version,
        use_cache=False,
        output_dir=str(export_path),
    )
    logger.info(f".blob model saved to {blob_path}")
    return Path(blob_path)


def hubai_export(
    cfg: HubAIExportConfig,
    quantization_mode: str,
    archive_path: PathType,
    export_path: PathType,
    model_name: str,
    dataset_name: str | None = None,
) -> Path:
    """Convert an ONNX NN Archive for a device through the HubAI SDK.

    The function uploads the archive to HubAI as a variant of the model
    named ``model_name``. It reuses the first model with this name, and
    creates the model when none exists. When the lookup of the models
    fails, the function logs a warning and creates a new model. The
    variant is named ``<model_name>:<dataset_name>``, or
    ``model_name`` when ``dataset_name`` is ``None`` or empty.

    ``cfg.platform`` selects the conversion call of the SDK: ``RVC3``
    for ``"rvc3"``, ``RVC4`` for ``"rvc4"``, and ``RVC2`` for
    ``"rvc2"`` or ``None``. The call receives the keyword arguments
    ``path``, ``quantization_mode``, ``name``, and ``model_id``. The
    entries of ``cfg.params`` go to the call too, and replace an
    argument of the same name.

    The SDK downloads the converted archive. The function moves it
    into ``export_path`` under its own file name. It then removes the
    download directory when that directory is empty and is not the
    working directory.

    When ``cfg.delete_remote_model`` is set, the function cleans up
    HubAI at the end:

    - It deletes the model that it created, also when the conversion
      raises an error.
    - It deletes only the new variant when the model existed before.
      This happens only when the conversion call returns.

    A failed deletion logs a warning and does not raise an error.

    Args:
        cfg (HubAIExportConfig): The ``exporter.hubai`` section of the
            config.
        quantization_mode (str): The precision to convert to, such as
            ``"INT8_STANDARD"`` or ``"FP16_STANDARD"``.
        archive_path (``PathType``): The ONNX NN Archive to convert.
        export_path (``PathType``): The directory that receives the
            converted archive.
        model_name (str): The name of the model on HubAI.
        dataset_name (str | None): The name of the train dataset. It
            is the second part of the variant name.

    Returns:
        ``Path``: The path of the converted archive inside
        ``export_path``.

    Raises:
        ValueError: When the ``HUBAI_API_KEY`` environment variable is
            not set or empty.
        NotImplementedError: When ``cfg.platform`` is ``"hailo"``.

    """
    from hubai_sdk import HubAIClient

    hubai_token = os.environ.get("HUBAI_API_KEY")
    if not hubai_token:
        raise ValueError(
            "HUBAI_API_KEY environment variable is not set. "
            "Please set it to use HubAI SDK for model conversion. "
        )

    client = HubAIClient(api_key=hubai_token)
    archive_path = Path(archive_path)

    existing_model_id = _find_existing_model_id(client, model_name)

    variant_name = (
        f"{model_name}:{dataset_name}" if dataset_name else f"{model_name}"
    )
    base_kwargs: dict = {
        "path": str(archive_path),
        "quantization_mode": quantization_mode,
        "name": variant_name,
    }

    base_kwargs["model_id"], created_model_id = _resolve_hubai_model(
        client, existing_model_id, model_name, variant_name
    )

    if cfg.params:
        base_kwargs.update(cfg.params)

    variant_id = None
    try:
        response = _convert_for_platform(client, cfg.platform, base_kwargs)
        variant_id = str(response.instance.model_version_id)
        return _finalize_hubai_output(response, export_path)
    finally:
        _cleanup_remote_model(client, cfg, created_model_id, variant_id)


def make_initializers_unique(onnx_path: PathType) -> None:
    """Give every node input its own copy of a shared ONNX initializer.

    The function counts how many node inputs of the main graph read
    each initializer. It replaces an initializer read by two or more
    inputs with one copy per input, named ``<name>_unique_<i>``. The
    ``i``-th such input in node order, from ``0``, reads the copy with
    index ``i``. An initializer read once, or not at all, keeps its
    name. The function saves the model over ``onnx_path`` and checks
    it with ``onnx.checker.check_model``. A failed check logs a
    warning. At the end, the function logs how many initializers it
    duplicated.

    When the graph has no initializers, the function logs a warning
    and leaves the file unchanged.

    Args:
        onnx_path (``PathType``): The ONNX file to modify.

    """
    import onnx

    onnx_path = str(onnx_path)
    model = onnx.load(onnx_path)
    graph = model.graph

    initializer_info = _collect_initializer_info(graph)
    if not initializer_info:
        logger.warning("No initializers found in the model")
        return

    name_mapping, new_initializers, duplicated_count = (
        _build_unique_initializers(initializer_info)
    )

    del graph.initializer[:]
    graph.initializer.extend(new_initializers)

    _remap_node_inputs(graph, name_mapping)

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


class _InitializerInfo(TypedDict):
    data: "TensorProto"
    usage_count: int


def _find_existing_model_id(
    client: "HubAIClient", model_name: str
) -> str | None:
    try:
        for model in client.models.list_models():
            if model.name == model_name:
                return str(model.id)
    except Exception as e:
        logger.warning(f"Failed to check for existing model: {e}")
    return None


def _resolve_hubai_model(
    client: "HubAIClient",
    existing_model_id: str | None,
    model_name: str,
    variant_name: str,
) -> tuple[str, str | None]:
    """Select the HubAI model of the variant, and create it when needed.

    The function selects ``existing_model_id`` when it is not ``None``.
    Otherwise it creates a model named ``model_name`` on HubAI. It logs
    which of the two happens.

    Args:
        client (``HubAIClient``): The client of the HubAI SDK.
        existing_model_id (str | None): The id of the model named
            ``model_name`` on HubAI, or ``None`` when no such model
            exists.
        model_name (str): The name of the model.
        variant_name (str): The name of the new variant, for the log
            message.

    Returns:
        tuple[str, str | None]: The id of the model that receives
        the variant, and the id of the model this call created. The
        second value is ``None`` when the model existed before.
        `hubai_export` deletes a whole model only when this call
        created it.

    """
    if existing_model_id is not None:
        logger.info(
            f"Model '{model_name}' already exists on HubAI. "
            f"Creating new variant '{variant_name}' under existing model."
        )
        return existing_model_id, None

    new_model = client.models.create_model(model_name)
    created_model_id = str(new_model.id)
    logger.info(
        f"Created new model '{model_name}' on HubAI. "
        f"Creating variant '{variant_name}' under it."
    )
    return created_model_id, created_model_id


def _convert_for_platform(
    client: "HubAIClient",
    platform: Literal["rvc2", "rvc3", "rvc4", "hailo"] | None,
    base_kwargs: dict,
) -> "ConvertResponse":
    # TODO: reintroduce Hailo conversion when modelconv is released
    # and hubai-sdk is updated accordingly
    if platform == "rvc3":
        return client.convert.RVC3(**base_kwargs)
    if platform == "rvc4":
        return client.convert.RVC4(**base_kwargs)
    if platform == "hailo":
        raise NotImplementedError(
            "Hailo platform conversion is not yet supported."
        )
    return client.convert.RVC2(**base_kwargs)


def _finalize_hubai_output(
    response: "ConvertResponse", export_path: PathType
) -> Path:
    downloaded_path = Path(response.downloaded_path)
    export_path = Path(export_path)
    output_path = export_path / downloaded_path.name
    downloaded_parent = downloaded_path.parent

    shutil.move(downloaded_path, output_path)

    if downloaded_parent.exists() and downloaded_parent != Path.cwd():
        with suppress(OSError):
            downloaded_parent.rmdir()

    logger.info(f"HubAI converted archive saved to {output_path}")
    return output_path


def _cleanup_remote_model(
    client: "HubAIClient",
    cfg: HubAIExportConfig,
    created_model_id: str | None,
    variant_id: str | None,
) -> None:
    if not cfg.delete_remote_model:
        return
    try:
        if created_model_id:
            client.models.delete_model(created_model_id)
            logger.debug(
                f"Cleaned up temporary HubAI model: {created_model_id}"
            )
        elif variant_id:
            client.variants.delete_variant(variant_id)
            logger.debug(f"Cleaned up temporary HubAI variant: {variant_id}")
    except Exception as e:
        resource_type = "model" if created_model_id else "variant"
        resource_id = created_model_id or variant_id
        logger.warning(
            f"Failed to cleanup HubAI {resource_type} '{resource_id}': {e}"
        )


def _collect_initializer_info(
    graph: "GraphProto",
) -> dict[str, _InitializerInfo]:
    initializer_info: dict[str, _InitializerInfo] = {
        initializer.name: {
            "data": copy.deepcopy(initializer),
            "usage_count": 0,
        }
        for initializer in graph.initializer
    }
    for node in graph.node:
        for input_name in node.input:
            if input_name in initializer_info:
                initializer_info[input_name]["usage_count"] += 1
    return initializer_info


def _build_unique_initializers(
    initializer_info: dict[str, _InitializerInfo],
) -> tuple[dict[str, list[str]], list["TensorProto"], int]:
    name_mapping: dict[str, list[str]] = defaultdict(list)
    new_initializers: list[TensorProto] = []
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

    return name_mapping, new_initializers, duplicated_count


def _remap_node_inputs(
    graph: "GraphProto", name_mapping: dict[str, list[str]]
) -> None:
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
