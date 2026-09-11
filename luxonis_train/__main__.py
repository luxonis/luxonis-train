"""The ``luxonis_train`` command line interface.

Every command builds a `LuxonisModel
<luxonis_train.core.core.LuxonisModel>` from a config and calls one of
its methods. ``--model`` and ``--variant`` select a packaged config, so
``--config`` is optional.

"""

import importlib
import importlib.util
import json
import sys
from collections.abc import Iterator
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal, TypeAlias, cast

import yaml
from cyclopts import App, Group, Parameter, validators
from loguru import logger
from luxonis_ml.typing import Params, PathType

from luxonis_train.upgrade import upgrade_config, upgrade_installation

_SECTION_BY_PACKAGE = {
    "backbones": "Backbone",
    "necks": "Neck",
    "heads": "Head",
}

if TYPE_CHECKING:
    import numpy as np
    from rich.console import Console

    from luxonis_train import LuxonisModel
    from luxonis_train.config import NodeConfig
    from luxonis_train.config.predefined_models import BasePredefinedModel
    from luxonis_train.loaders import BaseLoaderTorch

    OptsType: TypeAlias = list[str] | None
    LauncherToken: TypeAlias = str
    LauncherSource: TypeAlias = list[Path] | None
else:
    OptsType = Annotated[
        list[str] | None, Parameter(json_list=False, json_dict=False)
    ]
    LauncherToken = Annotated[
        str, Parameter(show=False, allow_leading_hyphen=True)
    ]
    LauncherSource = Annotated[
        list[Path] | None,
        Parameter(
            help="Path to a python module with custom components. "
            "This module will be sourced before running a command."
        ),
    ]


app = App(
    help="Luxonis Train CLI",
    version=lambda: f"LuxonisTrain v{version('luxonis_train')}",
)
app.meta.group_parameters = Group("Global Parameters", sort_key=0)
app["--help"].group = app.meta.group_parameters
app["--version"].group = app.meta.group_parameters

upgrade_app = app.command(App(name="upgrade"))

training_group = Group.create_ordered("Training")
evaluation_group = Group.create_ordered("Evaluation")
export_group = Group.create_ordered("Export")
annotation_group = Group.create_ordered("Annotation")
management_group = Group.create_ordered("Management")


def create_model(
    config: PathType | Params | None = None,
    opts: list[str] | None = None,
    weights: PathType | None = None,
    allow_empty_dataset: bool = False,
    *,
    model: str | None = None,
    variant: str | None = None,
) -> "LuxonisModel":
    importlib.reload(sys.modules["luxonis_train"])

    from luxonis_train import LuxonisModel

    return LuxonisModel(
        config,
        opts,
        model=model,
        variant=variant,
        weights=weights,
        allow_empty_dataset=allow_empty_dataset,
    )


@app.command(group=training_group, sort_key=1)
def train(
    opts: OptsType = None,
    /,
    *,
    config: str | None = None,
    model: str | None = None,
    variant: str | None = None,
    weights: str | None = None,
    debug: bool = False,
):
    """Start the training process.

    Args:
        opts (list[str]): A list of optional CLI overrides of the config file.
        config (str | None): ``Path`` to the configuration file.
        model (str | None): Name of a packaged predefined model, for example
            ``"detection"``. Mutually exclusive with ``config``. Run
            ``luxonis_train list-models`` to see the options.
        variant (str | None): Variant of the predefined model, for example
            ``"light"`` or ``"heavy"``. Defaults to the model's default variant.
        weights (str | None): ``Path`` to the model weights.
        debug (bool): If ``True``, allows the model to be constructed without
            a valid dataset by setting ``allow_empty_dataset`` to ``True``.

    """
    create_model(
        config,
        opts,
        weights=weights,
        allow_empty_dataset=debug,
        model=model,
        variant=variant,
    ).train(weights=weights)


@app.command(group=training_group, sort_key=2)
def tune(
    opts: OptsType = None,
    /,
    *,
    config: str | None = None,
    model: str | None = None,
    variant: str | None = None,
    weights: str | None = None,
    debug: bool = False,
):
    """Start hyperparameter tuning.

    Args:
        opts (list[str]): A list of optional CLI overrides of the config file.
        config (str | None): ``Path`` to the configuration file.
        model (str | None): Name of a packaged predefined model, for example
            ``"detection"``. Mutually exclusive with ``config``. Run
            ``luxonis_train list-models`` to see the options.
        variant (str | None): Variant of the predefined model, for example
            ``"light"`` or ``"heavy"``. Defaults to the model's default variant.
        weights (str | None): ``Path`` to the model weights.
        debug (bool): If ``True``, allows the model to be constructed without
            a valid dataset by setting ``allow_empty_dataset`` to ``True``.

    """
    create_model(
        config,
        opts,
        weights=weights,
        allow_empty_dataset=debug,
        model=model,
        variant=variant,
    ).tune()


@app.command(group=training_group, sort_key=3)
def inspect(
    opts: OptsType = None,
    /,
    *,
    config: str | None = None,
    model: str | None = None,
    variant: str | None = None,
    view: Literal["train", "val", "test"] = "train",
    size_multiplier: Annotated[
        float, Parameter(["--size_multiplier", "-s"])
    ] = 1.0,
    list_augmentations: bool = False,
):
    """Inspect the dataset as specified in the configuration.

    To close the window press ``"q"`` or ``"Esc"``.

    Args:
        opts (list[str]): A list of optional CLI overrides of the config file.
        config (str | None): ``Path`` to the configuration file.
        model (str | None): Name of a packaged predefined model, for example
            ``"detection"``. Mutually exclusive with ``config``. Run
            ``luxonis_train list-models`` to see the options.
        variant (str | None): Variant of the predefined model, for example
            ``"light"`` or ``"heavy"``. Defaults to the model's default variant.
        view (``Literal["train", "val", "test"]``): Dataset view to inspect.
        size_multiplier (float): Multiplier for the image size. By default,
            images are shown in their original size.
        list_augmentations (bool): Whether to show applied augmentations in the
            footer.

    """
    import cv2

    @lru_cache
    def get_window() -> str:
        window_name = "Visualization"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        return window_name

    for viz in _yield_visualizations(
        config=config,
        view=view,
        size_multiplier=size_multiplier,
        list_augmentations=list_augmentations,
        opts=opts,
        model=model,
        variant=variant,
    ):
        window_name = get_window()
        cv2.resizeWindow(window_name, width=viz.shape[1], height=viz.shape[0])
        cv2.imshow(window_name, viz)
        if cv2.waitKey() in {ord("q"), 27}:
            break
    cv2.destroyAllWindows()


@app.command(group=evaluation_group, sort_key=1)
def test(
    opts: OptsType = None,
    /,
    *,
    config: str | None = None,
    model: str | None = None,
    variant: str | None = None,
    view: Literal["train", "val", "test"] = "test",
    weights: str | None = None,
    debug: bool = False,
):
    """Evaluate a trained model.

    Args:
        opts (list[str]): A list of optional CLI overrides of the config file.
        config (str | None): ``Path`` to the configuration file or predefined model
            name.
        model (str | None): Name of a packaged predefined model, for example
            ``"detection"``. Mutually exclusive with ``config``. Run
            ``luxonis_train list-models`` to see the options.
        variant (str | None): Variant of the predefined model, for example
            ``"light"`` or ``"heavy"``. Defaults to the model's default variant.
        view (``Literal["train", "val", "test"]``): Dataset view to evaluate.
        weights (str | None): ``Path`` to the model weights.
        debug (bool): If ``True``, allows the model to be constructed without
            a valid dataset by setting ``allow_empty_dataset`` to ``True``.

    """
    create_model(
        config,
        opts,
        weights=weights,
        allow_empty_dataset=debug,
        model=model,
        variant=variant,
    ).test(view=view, weights=weights)


@app.command(group=evaluation_group, sort_key=2)
def infer(
    opts: OptsType = None,
    /,
    *,
    config: str | None = None,
    model: str | None = None,
    variant: str | None = None,
    view: Literal["train", "val", "test"] = "val",
    save_dir: Path | None = None,
    source_path: str | None = None,
    weights: str | None = None,
):
    """Run inference on a dataset view or a custom source.

    Supports both images and video files.

    Args:
        opts (list[str]): A list of optional CLI overrides of the config file.
        config (str | None): ``Path`` to the configuration file or predefined model
            name.
        model (str | None): Name of a packaged predefined model, for example
            ``"detection"``. Mutually exclusive with ``config``. Run
            ``luxonis_train list-models`` to see the options.
        variant (str | None): Variant of the predefined model, for example
            ``"light"`` or ``"heavy"``. Defaults to the model's default variant.
        view (``Literal["train", "val", "test"]``): Dataset view to use when
            ``source_path`` is not provided.
        save_dir (``Path | None``): Directory where inference results are saved.
        source_path (str | None): ``Path`` to an image file, image directory, or
            video file. If not provided, the loader from the configuration file
            is used.
        weights (str | None): ``Path`` to the model weights.

    """
    create_model(
        config,
        opts,
        weights=weights,
        allow_empty_dataset=True,
        model=model,
        variant=variant,
    ).infer(
        view=view,
        save_dir=save_dir,
        source_path=source_path,
        weights=weights,
    )


@app.command(group=annotation_group, sort_key=0)
def annotate(
    opts: OptsType = None,
    /,
    *,
    dir_path: Path,
    dataset_name: str,
    config: str | None = None,
    model: str | None = None,
    variant: str | None = None,
    weights: str | None = None,
    bucket_storage: Literal["local", "gcs"] = "local",
    delete_local: bool = True,
    delete_remote: bool = True,
    team_id: str | None = None,
):
    """Run annotation on a custom directory of images.

    Args:
        opts (list[str]): A list of optional CLI overrides of the config file.
        dir_path (``Path``): ``Path`` to the directory containing images to annotate.
        dataset_name (str): Name of the dataset for the annotated images.
        config (str | None): ``Path`` to the configuration file used by the model
            to annotate images.
        model (str | None): Name of a packaged predefined model, for example
            ``"detection"``. Mutually exclusive with ``config``. Run
            ``luxonis_train list-models`` to see the options.
        variant (str | None): Variant of the predefined model, for example
            ``"light"`` or ``"heavy"``. Defaults to the model's default variant.
        weights (str | None): ``Path`` to the model weights. If provided, the
            model uses these weights instead of those in the configuration
            file.
        bucket_storage (``Literal["local", "gcs"]``): Storage type for the new
            annotated dataset.
        delete_local (bool): Whether to delete the local dataset before
            writing.
        delete_remote (bool): Whether to delete the remote dataset before
            writing.
        team_id (str | None): Optional team ID for the dataset.

    """
    lx_model = create_model(
        config,
        opts,
        weights=weights,
        allow_empty_dataset=True,
        model=model,
        variant=variant,
    )

    lx_model.annotate(
        dir_path=dir_path,
        dataset_name=dataset_name,
        weights=weights,
        bucket_storage=bucket_storage,
        delete_local=delete_local,
        delete_remote=delete_remote,
        team_id=team_id,
    )


@app.command(group=export_group, sort_key=1)
def export(
    opts: OptsType = None,
    /,
    *,
    config: str | None = None,
    model: str | None = None,
    variant: str | None = None,
    save_path: str | None = None,
    weights: str | None = None,
    ckpt_only: bool = False,
):
    """Export the model to ONNX or BLOB format.

    Args:
        opts (list[str]): A list of optional CLI overrides of the config file.
        config (str | None): ``Path`` to the configuration file or predefined model
            name.
        model (str | None): Name of a packaged predefined model, for example
            ``"detection"``. Mutually exclusive with ``config``. Run
            ``luxonis_train list-models`` to see the options.
        variant (str | None): Variant of the predefined model, for example
            ``"light"`` or ``"heavy"``. Defaults to the model's default variant.
        save_path (str | None): Directory where exported model files are
            saved. If not specified, files are saved to the ``"export"``
            directory in the run save directory.
        weights (str | None): ``Path`` to the model weights.
        ckpt_only (bool): If ``True``, only the ``.ckpt`` file is exported.

    """
    create_model(
        config,
        opts,
        weights=weights,
        allow_empty_dataset=True,
        model=model,
        variant=variant,
    ).export(save_path=save_path, weights=weights, ckpt_only=ckpt_only)


@app.command(group=export_group, sort_key=2)
def archive(
    opts: OptsType = None,
    /,
    *,
    config: str | None = None,
    model: str | None = None,
    variant: str | None = None,
    executable: str | None = None,
    weights: str | None = None,
):
    """Convert the model to an NN Archive format.

    Args:
        opts (list[str]): A list of optional CLI overrides of the config file.
        config (str | None): ``Path`` to the configuration file.
        model (str | None): Name of a packaged predefined model, for example
            ``"detection"``. Mutually exclusive with ``config``. Run
            ``luxonis_train list-models`` to see the options.
        variant (str | None): Variant of the predefined model, for example
            ``"light"`` or ``"heavy"``. Defaults to the model's default variant.
        executable (str | None): ``Path`` to the exported model, usually an ONNX
            file. If not provided, the model is exported first.
        weights (str | None): ``Path`` to the model weights.

    """
    create_model(
        config,
        opts,
        weights=weights,
        allow_empty_dataset=True,
        model=model,
        variant=variant,
    ).archive(path=executable, weights=weights)


@app.command(group=export_group, sort_key=3)
def convert(
    opts: OptsType = None,
    /,
    *,
    config: str | None = None,
    model: str | None = None,
    variant: str | None = None,
    save_dir: str | None = None,
    weights: str | None = None,
):
    """Export, archive, and convert the model to target platform format.

    This is a unified command that combines export, archive, and
    platform conversion (RVC2/RVC3/RVC4) steps based on the
    configuration.

    Args:
        opts (list[str]): A list of optional CLI overrides of the config file.
        config (str | None): ``Path`` to the configuration file.
        model (str | None): Name of a packaged predefined model, for example
            ``"detection"``. Mutually exclusive with ``config``. Run
            ``luxonis_train list-models`` to see the options.
        variant (str | None): Variant of the predefined model, for example
            ``"light"`` or ``"heavy"``. Defaults to the model's default variant.
        save_dir (str | None): Directory where outputs are saved. If not
            specified, the default run save directory is used.
        weights (str | None): ``Path`` to the model weights.

    """
    create_model(
        config,
        opts,
        weights=weights,
        allow_empty_dataset=True,
        model=model,
        variant=variant,
    ).convert(save_dir=save_dir, weights=weights)


@app.command(group=export_group, sort_key=1)
def quantize(
    opts: list[str] | None = None,
    /,
    *,
    config: str | None = None,
    model: str | None = None,
    variant: str | None = None,
    weights: str | None = None,
):
    """Quantize the model using AIMET.

    Args:
        opts (list[str]): A list of optional CLI overrides of the config file.
        config (str | None): ``Path`` to the configuration file.
        model (str | None): Name of a packaged predefined model, for example
            ``"detection"``. Mutually exclusive with ``config``. Run
            ``luxonis_train list-models`` to see the options.
        variant (str | None): Variant of the predefined model, for example
            ``"light"`` or ``"heavy"``. Defaults to the model's default variant.
        weights (str | None): ``Path`` to the model weights.

    """
    lx_model = create_model(
        config,
        opts,
        weights=weights,
        allow_empty_dataset=False,
        model=model,
        variant=variant,
    )
    lx_model.quantize()


@app.command(group=management_group, sort_key=1, name="list-models")
def list_models():
    """List packaged predefined models, their variants and versions.

    Each row shows the model name, its variants and its versions. The
    ``*`` marks the default variant and version, which are used when the
    option is omitted.

    """
    from rich import box
    from rich.console import Console
    from rich.table import Table

    from luxonis_train.config.predefined import list_predefined_models

    entries = list_predefined_models()
    if not entries:
        Console().print("[yellow]No packaged predefined models found.[/]")
        return

    table = Table(
        title="Packaged predefined models",
        caption="[dim]* default when the option is omitted[/]",
        box=box.ROUNDED,
    )
    table.add_column("Model", style="bold cyan")
    table.add_column("Variants")
    table.add_column("Versions", style="green")
    for name, file_variants in entries.items():
        table.add_row(
            name, _variants_cell(name, file_variants[0]), _versions_cell(name)
        )

    Console().print(table)


@app.command(group=management_group, sort_key=2)
def info(*, model: str, variant: str | None = None):
    """Display documentation for a packaged predefined model.

    Args:
        model (str): Packaged model name, optionally suffixed with a version,
            for example ``"detection:v1"``.
        variant (str | None): Model variant to describe. Defaults to the
            packaged model's default variant.

    """
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text

    from luxonis_train.config.predefined import (
        parse_model_spec,
        resolve_predefined_config,
    )
    from luxonis_train.config.predefined_versions import (
        resolve_predefined_class,
        resolved_class_name,
    )

    importlib.import_module("luxonis_train.nodes")
    importlib.import_module("luxonis_train.config.predefined_models")

    family, requested_version = parse_model_spec(model)
    config = yaml.safe_load(
        resolve_predefined_config(family, variant).path.read_text()
    )
    predefined_config = config["model"]["predefined_model"]
    class_family = predefined_config["name"]
    version: int | str
    if requested_version is None or requested_version == "latest":
        version = "latest"
    else:
        version = int(requested_version)
    model_class = resolve_predefined_class(class_family, version)
    params = dict(predefined_config.get("params") or {})
    # The config layer allows `variant` both at the `predefined_model`
    # level and inside `params` (where it takes precedence).
    params_variant = params.pop("variant", None)
    selected_variant = (
        variant
        or params_variant
        or predefined_config.get("variant", "default")
    )
    predefined_model = cast(Any, model_class)(
        variant=selected_variant,
        **params,
    )
    resolved_name = resolved_class_name(class_family, version)

    console = Console()
    description = _docstring_to_text(model_class.__dict__.get("__doc__"))
    if not description:
        description = f"Predefined {class_family} architecture."
    console.print(
        Panel(
            Text(description),
            title=f"[bold]{family}[/] · {selected_variant} · {resolved_name}",
            border_style="cyan",
        )
    )

    node_configs = {node.name: node for node in predefined_model.nodes}
    for section, node_name in _info_components(predefined_model):
        if node_name is None:
            continue
        _print_node_panel(console, section, node_name, node_configs[node_name])


@upgrade_app.command()
def config(
    config: Annotated[
        Path,
        Parameter(validator=validators.Path(exists=True)),
        Parameter(validator=validators.Path(ext={"yaml", "yml", "json"})),
    ],
    output: Annotated[
        Path | None,
        Parameter(validator=validators.Path(ext={"yaml", "yml", "json"})),
    ] = None,
):
    """Upgrade luxonis-train configuration file.

    Args:
        config (``Path``): ``Path`` to configuration file to be upgraded.
        output (``Path | None``): Where to save the upgraded config. If omitted,
            the old file is overwritten.

    """
    new_cfg = upgrade_config(config)

    output = output or config
    if output.suffix == ".json":
        output.write_text(json.dumps(new_cfg, indent=2))
    else:
        with open(output, "w") as f:
            yaml.safe_dump(
                new_cfg, f, sort_keys=False, default_flow_style=False
            )


@upgrade_app.command(name=["checkpoint", "ckpt"])
def checkpoint(
    opts: OptsType = None,
    /,
    *,
    path: Annotated[
        Path,
        Parameter(validator=validators.Path(exists=True)),
    ],
    output: Path | None = None,
    config: Path | None = None,
):
    """Upgrade luxonis-train checkpoint file.

    Args:
        opts (list[str]): A list of optional CLI overrides of the config file.
        path (``Path``): ``Path`` to the checkpoint.
        output (``Path | None``): Where to save the upgraded checkpoint. If
            omitted, the old file is overwritten.
        config (``Path | None``): Optional configuration file used to construct the
            model.

    """
    from luxonis_train import LuxonisModel

    logger.info("Performing a full checkpoint upgrade.")
    model = LuxonisModel(config, opts, weights=path, allow_empty_dataset=True)
    model.lightning_module.load_checkpoint(path)

    # Needs to be called in order to attach the model to the trainer
    model.pl_trainer.validate(
        model.lightning_module,
        model.pytorch_loaders["val"],
        verbose=False,
    )
    model.pl_trainer.save_checkpoint(output or path, weights_only=False)
    logger.info(f"Saved upgraded checkpoint to '{output}'")


@upgrade_app.default()
def upgrade():
    """Upgrade luxonis-train installation and user files.

    Usage without a subcommand will trigger an upgrade of the
    ``luxonis-train`` PyPI package.

    """
    upgrade_installation()


@app.meta.default
def launcher(
    *tokens: LauncherToken,
    source: LauncherSource = None,
):
    if source:
        for src in source:
            spec = importlib.util.spec_from_file_location(src.stem, src)
            if spec:
                module = importlib.util.module_from_spec(spec=spec)
                if spec.loader:
                    spec.loader.exec_module(module)
    app(tokens)


def _get_visualization_item(
    loader: "BaseLoaderTorch", index: int
) -> tuple[dict[str, "np.ndarray"], dict[str, "np.ndarray"], list[str]]:
    import numpy as np
    from luxonis_ml.data.utils.cli_utils import get_tracked_augmentations

    raw_loader = getattr(loader, "loader", None)
    if raw_loader is not None:
        sample = raw_loader[index]
        images, labels = sample
        if isinstance(images, np.ndarray):
            images = {loader.image_source: images}
        remap_keypoints = getattr(loader, "_remap_keypoints", None)
        if (
            getattr(loader, "kpts_mapping_per_task", None) is not None
            and remap_keypoints is not None
        ):
            labels = remap_keypoints(labels)
        return (
            images,
            labels,
            list(get_tracked_augmentations(sample.metadata) or {}),
        )

    images, labels = loader[index]
    if not isinstance(images, dict):
        images = {loader.image_source: images}
    return (
        {
            name: image.numpy().transpose(1, 2, 0)
            for name, image in images.items()
        },
        {task: label.numpy() for task, label in labels.items()},
        [],
    )


def _yield_visualizations(
    opts: OptsType = None,
    config: str | None = None,
    view: Literal["train", "val", "test"] = "train",
    size_multiplier: Annotated[
        float, Parameter(["--size_multiplier", "-s"])
    ] = 1.0,
    list_augmentations: bool = False,
    *,
    model: str | None = None,
    variant: str | None = None,
) -> Iterator["np.ndarray"]:
    import cv2
    import numpy as np
    from luxonis_ml.data.utils.visualizations import (
        add_augmentation_footer,
        visualize,
    )

    from luxonis_train.utils.general import decode_text_metadata_labels

    opts = opts or []
    opts.extend(["trainer.preprocessing.normalize.active", "False"])

    lx_model = create_model(config, opts, model=model, variant=variant)

    loader = lx_model.loaders[view]

    metadata_types = loader.get_metadata_types()
    categorical_encodings = loader.get_categorical_encodings()
    for idx in range(len(loader)):
        np_images, np_labels, augmentations = _get_visualization_item(
            loader, idx
        )
        main_image = np_images[loader.image_source]
        main_image = cv2.cvtColor(main_image, cv2.COLOR_RGB2BGR).astype(
            np.uint8
        )
        np_labels = decode_text_metadata_labels(np_labels, metadata_types)

        h, w, _ = main_image.shape
        new_h, new_w = int(h * size_multiplier), int(w * size_multiplier)
        main_image = cv2.resize(main_image, (new_w, new_h))
        viz = visualize(
            image=main_image,
            labels=np_labels,
            classes=loader.get_classes(),
            source_name=loader.image_source,
            categorical_encodings=categorical_encodings,
        )
        if list_augmentations:
            viz = add_augmentation_footer(viz, augmentations)
        yield viz


def _variants_cell(name: str, default: str | None) -> str:
    from luxonis_train.config.predefined import list_variants

    labels = []
    for v in list_variants(name):
        label = v if v is not None else "<default>"
        labels.append(f"{label}*" if v == default else label)
    return ", ".join(labels)


def _versions_cell(name: str) -> str:
    from luxonis_train.config.predefined import class_family
    from luxonis_train.config.predefined_versions import list_versions

    family = class_family(name)
    versions = list_versions(family) if family else {}
    if not versions:
        return "-"
    latest = max(versions)
    return ", ".join(f"v{v}*" if v == latest else f"v{v}" for v in versions)


def _info_components(
    predefined_model: "BasePredefinedModel",
) -> tuple[tuple[str, str | None], ...]:
    from luxonis_train.config.predefined_models.base_predefined_model import (
        SimplePredefinedModel,
    )

    if isinstance(predefined_model, SimplePredefinedModel):
        return (
            ("Backbone", predefined_model._backbone),
            (
                "Neck",
                predefined_model._neck if predefined_model._use_neck else None,
            ),
            ("Head", predefined_model._head),
        )
    return tuple(
        (_node_section(node.name), node.name)
        for node in predefined_model.nodes
    )


def _node_section(node_name: str) -> str:
    from luxonis_train.registry import NODES

    module = NODES.get(node_name).__module__
    for package, label in _SECTION_BY_PACKAGE.items():
        if f".nodes.{package}." in module:
            return label
    return "Node"


def _print_node_panel(
    console: "Console",
    section: str,
    node_name: str,
    node_config: "NodeConfig",
) -> None:
    from rich.panel import Panel
    from rich.text import Text

    from luxonis_train.registry import NODES

    node_class = NODES.get(node_name)
    node_doc = (
        node_class.__dict__.get("__doc__") or node_class.__init__.__doc__
    )
    body = Text(_docstring_to_text(node_doc) or "No documentation available.")
    variant_label = node_config.variant or "default"
    console.print(
        Panel(
            body,
            title=f"[bold]{section}[/] · {node_name} ({variant_label})",
            border_style="green",
        )
    )


def _docstring_to_text(doc: str | None) -> str:
    """Flatten the reST markup of a docstring for terminal output."""
    import inspect
    import re

    text = inspect.cleandoc(doc or "")
    # reST comments and directives carry no meaning in a terminal; a
    # `code-block` keeps its literal body, which follows it indented.
    text = re.sub(
        r"^[ \t]*\.\. .*(?:\n[ \t]*\n)?", "", text, flags=re.MULTILINE
    )
    text = re.sub(r"`([^`<]+?)(\s*)<([^>]+)>`_", r"\1\2(\3)", text)
    text = re.sub(r":\w+:`([^`]+)`", r"\1", text)
    text = text.replace("``", "")
    return re.sub(r"`([^`]+)`_?", r"\1", text)


if __name__ == "__main__":
    app.meta()
