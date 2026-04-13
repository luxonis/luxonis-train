import importlib
import importlib.util
import json
import sys
from collections.abc import Iterator
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

import yaml
from cyclopts import App, Group, Parameter, validators
from loguru import logger
from luxonis_ml.typing import Params, PathType

from luxonis_train.upgrade import upgrade_config, upgrade_installation

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


def create_model(
    config: PathType | Params | None,
    opts: list[str] | None = None,
    weights: PathType | None = None,
    allow_empty_dataset: bool = False,
) -> "LuxonisModel":
    importlib.reload(sys.modules["luxonis_train"])

    from luxonis_train import LuxonisModel

    return LuxonisModel(
        config,
        opts,
        weights=weights,
        allow_empty_dataset=allow_empty_dataset,
    )


@app.command(group=training_group, sort_key=1)
def train(
    opts: list[str] | None = None,
    /,
    *,
    config: str | None = None,
    weights: str | None = None,
    debug: bool = False,
):
    """Start the training process.

    @type config: str
    @param config: Path to the configuration file.
    @type weights: str
    @param weights: Path to the model weights.
    @type opts: list[str]
    @param opts: A list of optional CLI overrides of the config file.
    @type debug: bool
    @param debug: If True, the training will run in debug mode which
        suppresses some exceptions to allow training without a fully
        defined model.
    """
    create_model(
        config, opts, weights=weights, allow_empty_dataset=debug
    ).train(weights=weights)


@app.command(group=training_group, sort_key=2)
def tune(
    opts: list[str] | None = None,
    /,
    *,
    config: str | None = None,
    weights: str | None = None,
    debug: bool = False,
):
    """Start hyperparameter tuning.

    @type config: str
    @param config: Path to the configuration file.
    @type opts: list[str]
    @param opts: A list of optional CLI overrides of the config file.
    @type weights: str
    @param weights: Path to the model weights.
    @type debug: bool
    @param debug: If True, the tuning will run in debug mode which
        suppresses some exceptions to allow tuning without a fully
        defined model.
    """
    create_model(
        config, opts, weights=weights, allow_empty_dataset=debug
    ).tune()


def _yield_visualizations(
    opts: list[str] | None = None,
    config: str | None = None,
    view: Literal["train", "val", "test"] = "train",
    size_multiplier: Annotated[
        float, Parameter(["--size_multiplier", "-s"])
    ] = 1.0,
) -> Iterator["np.ndarray"]:
    import cv2
    import numpy as np
    from luxonis_ml.data.utils.visualizations import visualize

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
        return (
            {
                name: image.numpy().transpose(1, 2, 0)
                for name, image in images.items()
            },
            {task: label.numpy() for task, label in labels.items()},
        )

    opts = opts or []
    opts.extend(["trainer.preprocessing.normalize.active", "False"])

    model = create_model(config, opts)

    loader = model.loaders[view]
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
        yield visualize(
            image=main_image,
            labels=np_labels,
            classes=loader.get_classes(),
            source_name=loader.image_source,
            categorical_encodings=categorical_encodings,
        )


@app.command(group=training_group, sort_key=3)
def inspect(
    opts: list[str] | None = None,
    /,
    *,
    config: str | None = None,
    view: Literal["train", "val", "test"] = "train",
    size_multiplier: Annotated[
        float, Parameter(["--size_multiplier", "-s"])
    ] = 1.0,
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
        opts=opts,
    ):
        window_name = get_window()
        cv2.resizeWindow(window_name, width=viz.shape[1], height=viz.shape[0])
        cv2.imshow(window_name, viz)
        if cv2.waitKey() in {ord("q"), 27}:
            break
    cv2.destroyAllWindows()


@app.command(group=evaluation_group, sort_key=1)
def test(
    opts: list[str] | None = None,
    /,
    *,
    config: str | None = None,
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
    @param debug: If True, the training will run in debug mode which
        suppresses some exceptions to allow training without a fully
        defined model.
    """
    create_model(
        config, opts, weights=weights, allow_empty_dataset=debug
    ).test(view=view, weights=weights)


@app.command(group=evaluation_group, sort_key=2)
def infer(
    opts: list[str] | None = None,
    /,
    *,
    config: str | None = None,
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
        config, opts, weights=weights, allow_empty_dataset=True
    ).infer(
        view=view,
        save_dir=save_dir,
        source_path=source_path,
        weights=weights,
    )


@app.command(group=annotation_group, sort_key=0)
def annotate(
    opts: list[str] | None = None,
    /,
    *,
    dir_path: Path,
    dataset_name: str,
    config: str | None = None,
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
    model = create_model(
        config, opts, weights=weights, allow_empty_dataset=True
    )

    model.annotate(
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
    opts: list[str] | None = None,
    /,
    *,
    config: str | None = None,
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
        This is useful for updating the metadata in the checkpoint
        file in case they changed (e.g. new configuration file,
        architectural changes affecting the exection order etc.)
    @type weights: str
    @param weights: Path to the model weights.
    @type opts: list[str]
    @param opts: A list of optional CLI overrides of the
    """
    create_model(
        config, opts, weights=weights, allow_empty_dataset=True
    ).export(save_path=save_path, weights=weights, ckpt_only=ckpt_only)


@app.command(group=export_group, sort_key=2)
def archive(
    opts: list[str] | None = None,
    /,
    *,
    config: str | None,
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
        config, opts, weights=weights, allow_empty_dataset=True
    ).archive(path=executable, weights=weights)


@app.command(group=export_group, sort_key=3)
def convert(
    opts: list[str] | None = None,
    /,
    *,
    config: str | None = None,
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
        config, opts, weights=weights, allow_empty_dataset=True
    ).convert(save_dir=save_dir, weights=weights)


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
    opts: list[str] | None = None,
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
