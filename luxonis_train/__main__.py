import importlib
import importlib.util
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

import requests
from cyclopts import App, Group, Parameter

if TYPE_CHECKING:
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


def create_model(config: str | None, opts: list[str] | None) -> "LuxonisModel":
    importlib.reload(sys.modules["luxonis_train"])
    from luxonis_train import LuxonisModel

    return LuxonisModel(config, opts)


@app.command(group=training_group, sort_key=1)
def train(
    opts: list[str] | None = None,
    /,
    *,
    config: str | None = None,
    resume_weights: str | None = None,
):
    """Start the training process.

    @type config: str
    @param config: Path to the configuration file.
    @type resume_weights: str
    @param resume_weights: Path to the model weights to resume training
        from. @type *opts: tuple[str, str]
    @param opts: A list of optional CLI overrides of the config file.
    """
    create_model(config, opts).train(resume_weights=resume_weights)


@app.command(group=training_group, sort_key=2)
def tune(opts: list[str] | None = None, /, *, config: str | None = None):
    """Start hyperparameter tuning.

    @type config: str
    @param config: Path to the configuration file.
    @type opts: list[str]
    @param opts: A list of optional CLI overrides of the config file.
    """
    create_model(config, opts).tune()


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
        viz = visualize(
            main_image,
            np_labels,
            loader.get_classes(),
        )
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
    view: Literal["train", "val", "test"] = "val",
    weights: str | None = None,
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
    """
    create_model(config, opts).test(view=view, weights=weights)


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


@app.command(group=export_group, sort_key=1)
def export(
    opts: list[str] | None = None,
    /,
    *,
    config: str | None = None,
    save_path: str | None = None,
    weights: str | None = None,
):
    """Export the model to ONNX or BLOB format.

    @type config: str
    @param config: Path to the configuration file or a name of a
        predefined model.
    @type save_path: str
    @param save_path: Path to save the exported model.
    @type weights: str
    @param weights: Path to the model weights.
    @type opts: list[str]
    @param opts: A list of optional CLI overrides of the
    """
    create_model(config, opts).export(
        onnx_save_path=save_path, weights=weights
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
        response = requests.get(url)
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
        Path | None,
        Parameter(
            help="Path to a python module with custom components. "
            "This module will be sourced before running a command."
        ),
    ] = None,
):
    if source:
        spec = importlib.util.spec_from_file_location(source.stem, source)
        if spec:
            module = importlib.util.module_from_spec(spec=spec)
            if spec.loader:
                spec.loader.exec_module(module)
    app(tokens)


if __name__ == "__main__":
    app.meta()
