import importlib
import importlib.util
import json
import sys
from collections.abc import Iterator
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal, TypeAlias

import yaml
from cyclopts import App, Group, Parameter, validators
from loguru import logger
from luxonis_ml.typing import Params, PathType

from luxonis_train.config.predefined import (
    list_predefined_models,
    resolve_predefined_config,
)
from luxonis_train.upgrade import upgrade_config, upgrade_installation

OptsType: TypeAlias = Annotated[
    list[str] | None, Parameter(json_list=False, json_dict=False)
]

if TYPE_CHECKING:
    import numpy as np

    from luxonis_train import LuxonisModel


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


def _split_model_version(model: str) -> tuple[str, str | None]:
    """Split ``family:vN`` (or ``family:latest``) into ``(family,
    version)``.

    Returns the version portion unchanged (as a string) so callers can
    tell "no colon" from "``:latest``". The version is not parsed to an
    int here — it is passed straight through as a config override.
    """
    if ":" not in model:
        return model, None
    family, _, version_str = model.partition(":")
    if version_str.startswith("v") and version_str[1:].isdigit():
        return family, version_str[1:]
    if version_str == "latest":
        return family, "latest"
    raise ValueError(
        f"Malformed model spec '{model}'. Expected '<name>', "
        f"'<name>:vN' (e.g. detection:v1), or '<name>:latest'."
    )


def _resolve_config(
    config: PathType | Params | None,
    model: str | None,
    variant: str | None,
) -> PathType | Params | None:
    if model is None:
        if variant is not None:
            raise ValueError(
                "'--variant' requires '--model' to be specified as well."
            )
        return config
    if config is not None:
        raise ValueError("'--config' and '--model' are mutually exclusive.")
    family, _ = _split_model_version(model)
    return str(resolve_predefined_config(family, variant))


def create_model(
    config: PathType | Params | None,
    opts: list[str] | None = None,
    weights: PathType | None = None,
    allow_empty_dataset: bool = False,
    *,
    model: str | None = None,
    variant: str | None = None,
) -> "LuxonisModel":
    if model is not None:
        _, version = _split_model_version(model)
        if version is not None and version != "latest":
            opts = [
                *list(opts or []),
                "model.predefined_model.version",
                version,
            ]
    resolved = _resolve_config(config, model, variant)

    importlib.reload(sys.modules["luxonis_train"])

    from luxonis_train import LuxonisModel

    return LuxonisModel(
        resolved,
        opts,
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
    from luxonis_ml.data.utils.augmentations_collector import (
        AugmentationsCollector,
    )
    from luxonis_ml.data.utils.visualizations import (
        add_augmentation_footer,
        visualize,
    )

    from luxonis_train.utils.general import decode_text_metadata_labels

    def get_visualization_item(
        idx: int,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        raw_loader = getattr(loader, "loader", None)
        if raw_loader is not None:
            np_images, np_labels = raw_loader[idx]
            if isinstance(np_images, np.ndarray):
                np_images = {loader.image_source: np_images}

            remap_keypoints = getattr(loader, "_remap_keypoints", None)
            if (
                getattr(loader, "kpts_mapping_per_task", None) is not None
                and remap_keypoints is not None
            ):
                np_labels = remap_keypoints(np_labels)

            return np_images, np_labels

        images, labels = loader[idx]
        if not isinstance(images, dict):
            images = {loader.image_source: images}
        return (
            {
                name: image.numpy().transpose(1, 2, 0)
                for name, image in images.items()
            },
            {task: label.numpy() for task, label in labels.items()},
        )

    opts = opts or []
    opts.extend(["trainer.preprocessing.normalize.active", "False"])

    lx_model = create_model(config, opts, model=model, variant=variant)

    loader = lx_model.loaders[view]
    raw_loader = getattr(loader, "loader", None)
    if list_augmentations and raw_loader is not None:
        collector = AugmentationsCollector(
            raw_loader.augmentations,  # type: ignore[attr-defined]
            [
                aug.model_dump()
                for aug in lx_model.cfg_preprocessing.get_active_augmentations()
            ],
        )
        get_applied_augmentations = collector.get_applied_augmentations
    else:
        get_applied_augmentations = list

    metadata_types = loader.get_metadata_types()
    categorical_encodings = loader.get_categorical_encodings()
    for idx in range(len(loader)):
        np_images, np_labels = get_visualization_item(idx)
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
            viz = add_augmentation_footer(viz, get_applied_augmentations())
        yield viz


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
    @param config: Path to the configuration file.
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
    @param config: Path to the configuration file or a name of a
        predefined model.
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
    @param config: Path to the configuration file or a name of a
        predefined model.
    @type view: str
    @param view: Which dataset view to use. Only relevant when the
        source_path is not provided.
    @type save_dir: Path
    @param save_dir: Where to save the inference results.
    @type source_path: str
    @param source_path: Path to an image file, a directory containing
        images or a video file for inference. If not provided, the
        loader from the configuation file will be used.
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
        annotate images.
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
    @param config: Path to the configuration file or a name of a
        predefined model.
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
    @param config: Path to the configuration file.
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
    @param config: Path to the configuration file.
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
    from importlib.resources import files

    from rich import box
    from rich.console import Console
    from rich.table import Table

    from luxonis_train.config.predefined import _resolve_filename
    from luxonis_train.config.predefined_versions import list_versions

    entries = list_predefined_models()
    if not entries:
        Console().print("[yellow]No packaged predefined models found.[/]")
        return

    def _load_class_family(model: str) -> str | None:
        filename = _resolve_filename(model, None)
        try:
            data = yaml.safe_load(
                Path(str(files("configs") / filename)).read_text()
            )
            return data["model"]["predefined_model"]["name"]
        except Exception:
            return None

    table = Table(
        title="Packaged predefined models",
        caption="[dim]* default when the option is omitted[/]",
        box=box.ROUNDED,
    )
    table.add_column("Model", style="bold cyan")
    table.add_column("Variants")
    table.add_column("Versions", style="green")
    for name, variants in entries.items():
        default = variants[0]
        rendered_variants = []
        for v in variants:
            label = v if v is not None else "<default>"
            rendered_variants.append(f"{label}*" if v == default else label)
        class_family = _load_class_family(name)
        if class_family is not None:
            versions = list_versions(class_family)
            if versions:
                latest = max(versions)
                version_str = ", ".join(
                    f"v{v}*" if v == latest else f"v{v}" for v in versions
                )
            else:
                version_str = "-"
        else:
            version_str = "?"
        table.add_row(name, ", ".join(rendered_variants), version_str)

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

    from luxonis_train.config.predefined_models.base_predefined_model import (
        SimplePredefinedModel,
    )
    from luxonis_train.config.predefined_versions import (
        resolve_predefined_class,
        resolved_class_name,
    )
    from luxonis_train.registry import NODES

    importlib.import_module("luxonis_train.nodes")
    importlib.import_module("luxonis_train.config.predefined_models")

    family, requested_version = _split_model_version(model)
    config_path = resolve_predefined_config(family, variant)
    config = yaml.safe_load(config_path.read_text())
    predefined_config = config["model"]["predefined_model"]
    class_family = predefined_config["name"]
    version: int | str = (
        int(requested_version)
        if requested_version not in {None, "latest"}
        else "latest"
    )
    model_class = resolve_predefined_class(class_family, version)
    selected_variant = variant or predefined_config.get("variant", "default")
    predefined_model = model_class(
        variant=selected_variant,
        **predefined_config.get("params", {}),
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

    if not isinstance(predefined_model, SimplePredefinedModel):
        console.print(
            "[yellow]Component documentation is available for "
            "SimplePredefinedModel presets only.[/]"
        )
        return

    node_configs = {node.name: node for node in predefined_model.nodes}
    components = (
        ("Backbone", predefined_model._backbone),
        (
            "Neck",
            predefined_model._neck if predefined_model._use_neck else None,
        ),
        ("Head", predefined_model._head),
    )
    for section, node_name in components:
        if node_name is None:
            continue
        node_config = node_configs[node_name]
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
        old file will be overriden.
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
    @param new: Where to save the upgraded checkpoint. If left empty,
        the old file will be overriden.
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


if __name__ == "__main__":
    app.meta()
