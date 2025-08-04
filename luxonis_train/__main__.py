import importlib
import importlib.util
import subprocess
import sys
from collections.abc import Iterator
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

import requests
from cyclopts import App, Group, Parameter

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

training_group = Group.create_ordered("Training")
evaluation_group = Group.create_ordered("Evaluation")
export_group = Group.create_ordered("Export")
management_group = Group.create_ordered("Management")
annotation_group = Group.create_ordered("Annotation")


def create_model(
    config: str | None, opts: list[str] | None, debug_mode: bool = False
) -> "LuxonisModel":
    importlib.reload(sys.modules["luxonis_train"])
    from luxonis_train import LuxonisModel

    return LuxonisModel(config, opts, debug_mode=debug_mode)


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
    create_model(config, opts, debug).train(weights=weights)


@app.command(group=training_group, sort_key=2)
def tune(opts: list[str] | None = None, /, *, config: str | None = None):
    """Start hyperparameter tuning.

    @type config: str
    @param config: Path to the configuration file.
    @type opts: list[str]
    @param opts: A list of optional CLI overrides of the config file.
    """
    create_model(config, opts).tune()


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

    opts = opts or []
    opts.extend(["trainer.preprocessing.normalize.active", "False"])

    model = create_model(config, opts)

    loader = model.loaders[view]
    for images, labels in loader:
        np_images = {
            k: v.numpy().transpose(1, 2, 0) for k, v in images.items()
        }
        main_image = np_images[loader.image_source]
        main_image = cv2.cvtColor(main_image, cv2.COLOR_RGB2BGR).astype(
            np.uint8
        )
        np_labels = {task: label.numpy() for task, label in labels.items()}

        h, w, _ = main_image.shape
        new_h, new_w = int(h * size_multiplier), int(w * size_multiplier)
        main_image = cv2.resize(main_image, (new_w, new_h))
        yield visualize(
            image=main_image,
            labels=np_labels,
            classes=loader.get_classes(),
            source_name=loader.image_source,
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

    for viz in _yield_visualizations(
        config=config,
        view=view,
        size_multiplier=size_multiplier,
        opts=opts,
    ):
        cv2.imshow("Visualization", viz)
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
    create_model(config, opts, debug).test(view=view, weights=weights)


@app.command(group=evaluation_group, sort_key=2)
def infer(
    opts: list[str] | None = None,
    /,
    *,
    config: str | None = None,
    view: Literal["train", "val", "test"] = "val",
    save_dir: Path | None = None,
    source_path: str | None = None,
    weights: Path | None = None,
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
    create_model(config, opts).infer(
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
    config: str,
    dir_path: Path,
    dataset_name: str,
    weights: Path | None = None,
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
    model = create_model(config, opts)

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
    create_model(config, opts).export(
        save_path=save_path, weights=weights, ckpt_only=ckpt_only
    )


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
    create_model(str(config), opts).archive(path=executable, weights=weights)


@app.command(group=management_group)
def upgrade():
    """Update LuxonisTrain to the latest stable version."""

    def get_latest_version() -> str | None:
        url = "https://pypi.org/pypi/luxonis_train/json"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            versions = list(data["releases"].keys())
            versions.sort(key=lambda s: [int(u) for u in s.split(".")])
            return versions[-1]
        return None

    current_version = version("luxonis_train")
    latest_version = get_latest_version()
    if latest_version is None:
        print("Failed to check for updates. Try again later.")
        return
    if current_version == latest_version:
        print(f"LuxonisTrain is up-to-date (v{current_version}).")
    else:
        subprocess.check_output(
            f"{sys.executable} -m pip install -U pip".split()
        )
        subprocess.check_output(
            f"{sys.executable} -m pip install -U luxonis_train".split()
        )
        print(
            f"LuxonisTrain updated from v{current_version} to v{latest_version}."
        )


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
