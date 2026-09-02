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

OptsType: TypeAlias = Annotated[
    list[str] | None, Parameter(json_list=False, json_dict=False)
]

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

    @type config: str
    @param config: Path to the configuration file. Mutually exclusive
        with `--model`.
    @type model: str
    @param model: Name of a packaged predefined model (e.g.
        `detection`). Run `luxonis_train list-models` to see the
        options.
    @type variant: str
    @param variant: Variant of the predefined model (e.g. `light`,
        `heavy`). Defaults to the model's default variant.
    @type weights: str
    @param weights: Path to the model weights.
    @type opts: list[str]
    @param opts: A list of optional CLI overrides of the config file.
    @type debug: bool
    @param debug: If true, allows the model to be constructed without a
        valid dataset by setting `allow_empty_dataset` to True. This can
        be useful for quick testing of the training loop.
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

    @type config: str
    @param config: Path to the configuration file. Mutually exclusive
        with `--model`.
    @type model: str
    @param model: Name of a packaged predefined model.
    @type variant: str
    @param variant: Variant of the predefined model.
    @type opts: list[str]
    @param opts: A list of optional CLI overrides of the config file.
    @type weights: str
    @param weights: Path to the model weights.
    @type debug: bool
    @param debug: If true, allows the model to be constructed without a
        valid dataset by setting `allow_empty_dataset` to True. This can
        be useful for quick testing of the tuning.
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

    To close the window press 'q' or 'Esc'.

    @type config: str
    @param config: Path to the configuration file. Mutually exclusive
        with `--model`.
    @type model: str
    @param model: Name of a packaged predefined model (e.g.
        `detection`). Mutually exclusive with `--config`. Run
        `luxonis_train list-models` to see the options.
    @type variant: str
    @param variant: Variant of the predefined model (e.g. `light`,
        `heavy`). Defaults to the model's default variant.
    @type view: Literal["train", "val", "test"]
    @param view: Which dataset view to use. Only relevant when the
        source_path is not provided.
    @type size_multiplier: float
    @param size_multiplier: Multiplier for the image size. By default
        the images are shown in their original size. Use this option to
        scale them.
    @type list_augmentations: bool
    @param list_augmentations: Show the augmentations applied to each
        displayed image in the footer.
    @type opts: list[str]
    @param opts: A list of optional CLI overrides of the config file.
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

    @type config: str
    @param config: Path to the configuration file. Mutually exclusive
        with `--model`.
    @type model: str
    @param model: Name of a packaged predefined model (e.g.
        `detection`). Mutually exclusive with `--config`. Run
        `luxonis_train list-models` to see the options.
    @type variant: str
    @param variant: Variant of the predefined model (e.g. `light`,
        `heavy`). Defaults to the model's default variant.
    @type view: str
    @param view: Which dataset view to use. Only relevant when the
        source_path is not provided.
    @type weights: str
    @param weights: Path to the model weights.
    @type opts: list[str]
    @param opts: A list of optional CLI overrides of the config file.
    @type debug: bool
    @param debug: If true, allows the model to be constructed without a
        valid dataset by setting `allow_empty_dataset` to True. This can
        be useful for quick testing of the evaluation loop.
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

    @type config: str
    @param config: Path to the configuration file. Mutually exclusive
        with `--model`.
    @type model: str
    @param model: Name of a packaged predefined model (e.g.
        `detection`). Mutually exclusive with `--config`. Run
        `luxonis_train list-models` to see the options.
    @type variant: str
    @param variant: Variant of the predefined model (e.g. `light`,
        `heavy`). Defaults to the model's default variant.
    @type view: str
    @param view: Which dataset view to use. Only relevant when the
        source_path is not provided.
    @type save_dir: Path
    @param save_dir: Where to save the inference results.
    @type source_path: str
    @param source_path: Path to an image file, a directory containing
        images or a video file for inference. If not provided, the
        loader from the configuration file will be used.
    @type weights: Path
    @param weights: Path to the model weights.
    @type opts: list[str]
    @param opts: A list of optional CLI overrides of the config file.
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

    @type config: str
    @param config: Path to the configuration file used by the model to
        annotate images. Mutually exclusive with `--model`.
    @type model: str
    @param model: Name of a packaged predefined model (e.g.
        `detection`). Mutually exclusive with `--config`. Run
        `luxonis_train list-models` to see the options.
    @type variant: str
    @param variant: Variant of the predefined model (e.g. `light`,
        `heavy`). Defaults to the model's default variant.
    @type dir_path: str
    @param dir_path: Path to the directory containing images to
        annotate.
    @type dataset_name: str
    @param dataset_name: Name of the dataset for the annotated images.
    @type weights: Path | None
    @param weights: Path to the model weights. If provided, the model
        will use these weights instead of those in the configuration
        file.
    @type bucket_storage: Literal["local", "gcs"]
    @param bucket_storage: Storage type for the new annotated dataset.
    @type delete_local: bool
    @param delete_local: Whether to delete local dataset or append data
        to existing dataset.
    @type delete_remote: bool
    @param delete_remote: Whether to delete remote dataset or append
        data to existing dataset.
    @type team_id: str | None
    @param team_id: Optional team ID for the dataset.
    @type opts: list[str]
    @param opts: A list of optional CLI overrides of the config file.
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

    @type config: str
    @param config: Path to the configuration file. Mutually exclusive
        with `--model`.
    @type model: str
    @param model: Name of a packaged predefined model (e.g.
        `detection`). Mutually exclusive with `--config`. Run
        `luxonis_train list-models` to see the options.
    @type variant: str
    @param variant: Variant of the predefined model (e.g. `light`,
        `heavy`). Defaults to the model's default variant.
    @type save_path: str
    @param save_path: Directory where to save all exported model files.
        If not specified, files will be saved to the 'export' directory
        in the run save directory.
    @type ckpt_only: bool
    @param ckpt_only: If True, only the `.ckpt` file will be exported.
        This is useful for updating the metadata in the checkpoint file
        in case they changed (e.g. new configuration file, architectural
        changes affecting the execution order etc.)
    @type weights: str
    @param weights: Path to the model weights.
    @type opts: list[str]
    @param opts: A list of optional CLI overrides of the
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

    @type config: str
    @param config: Path to the configuration file. Mutually exclusive
        with `--model`.
    @type model: str
    @param model: Name of a packaged predefined model (e.g.
        `detection`). Mutually exclusive with `--config`. Run
        `luxonis_train list-models` to see the options.
    @type variant: str
    @param variant: Variant of the predefined model (e.g. `light`,
        `heavy`). Defaults to the model's default variant.
    @type executable: str
    @param executable: Path to the exported model, usually an ONNX file.
        If not provided, the model will be exported first.
    @type weights: str
    @param weights: Path to the model weights.
    @type opts: list[str]
    @param opts: A list of optional CLI overrides of the config file.
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

    @type config: str
    @param config: Path to the configuration file. Mutually exclusive
        with `--model`.
    @type model: str
    @param model: Name of a packaged predefined model (e.g.
        `detection`). Mutually exclusive with `--config`. Run
        `luxonis_train list-models` to see the options.
    @type variant: str
    @param variant: Variant of the predefined model (e.g. `light`,
        `heavy`). Defaults to the model's default variant.
    @type save_dir: str
    @param save_dir: Directory where all outputs will be saved. If not
        specified, the default run save directory will be used.
    @type weights: str
    @param weights: Path to the model weights.
    @type opts: list[str]
    @param opts: A list of optional CLI overrides of the config file.
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

    @type config: str
    @param config: Path to the configuration file.
    @type model: str
    @param model: Name of a packaged predefined model.
    @type variant: str
    @param variant: Variant of the predefined model.
    @type weights: str
    @param weights: Path to the model weights.
    @type opts: list[str]
    @param opts: A list of optional CLI overrides of the config file.
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

    Each row shows `<model>  variants  versions`. The `*` marks the
    default variant / version picked when the option is omitted.
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

    @type model: str
    @param model: Packaged model name, optionally suffixed with a
        version (for example `detection:v1`).
    @type variant: str | None
    @param variant: Model variant to describe. Defaults to the packaged
        model's default variant.
    """
    import inspect

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
    description = inspect.cleandoc(model_class.__dict__.get("__doc__") or "")
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

    @type config: Path
    @param config: Path to configuration file to be upgraded.
    @type output: Path | None
    @param output: Where to save the upgraded config. If left empty, the
        old file will be overridden.
    """
    if config.suffix == "json":
        cfg = json.loads(config.read_text(encoding="utf-8"))
    else:
        cfg = yaml.safe_load(config.read_text(encoding="utf-8"))

    new_cfg = upgrade_config(cfg)

    output = output or config
    if output.suffix == "json":
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

    @type path: Path
    @param path: Path to the checkpoint
    @type output: Path | None
    @param output: Where to save the upgraded checkpoint. If left empty,
        the old file will be overridden.
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

    Usage without a subcommand will trigger an upgrade of `luxonis-
    train` PyPI package.
    """
    upgrade_installation()


@app.meta.default
def launcher(
    *tokens: Annotated[str, Parameter(show=False, allow_leading_hyphen=True)],
    source: Annotated[
        list[Path] | None,
        Parameter(
            help="Path to a python module with custom components. "
            "This module will be sourced before running a command."
        ),
    ] = None,
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
    import inspect

    from rich.panel import Panel
    from rich.text import Text

    from luxonis_train.registry import NODES

    node_class = NODES.get(node_name)
    node_doc = (
        node_class.__dict__.get("__doc__") or node_class.__init__.__doc__
    )
    node_doc = inspect.cleandoc(node_doc or "")
    body = Text(node_doc or "No documentation available.")
    variant_label = node_config.variant or "default"
    console.print(
        Panel(
            body,
            title=f"[bold]{section}[/] · {node_name} ({variant_label})",
            border_style="green",
        )
    )


if __name__ == "__main__":
    app.meta()
