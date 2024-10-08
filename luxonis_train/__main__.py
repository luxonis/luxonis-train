from enum import Enum
from importlib.metadata import version
from pathlib import Path
from typing import Annotated

import typer
from luxonis_ml.utils import setup_logging

from luxonis_train.config import Config

setup_logging(use_rich=True)


class _ViewType(str, Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


app = typer.Typer(
    help="Luxonis Train CLI",
    add_completion=False,
    pretty_exceptions_show_locals=False,
)


ConfigType = Annotated[
    str | None,
    typer.Option(
        help="Path to the configuration file.",
        show_default=False,
        metavar="FILE",
    ),
]

OptsType = Annotated[
    list[str] | None,
    typer.Argument(
        help="A list of optional CLI overrides of the config file.",
        show_default=False,
    ),
]

WeightsType = Annotated[
    Path | None,
    typer.Option(
        help="Path to the model weights.",
        show_default=False,
        metavar="FILE",
    ),
]

ViewType = Annotated[
    _ViewType, typer.Option(help="Which dataset view to use.")
]

SaveDirType = Annotated[
    Path | None,
    typer.Option(help="Where to save the inference results."),
]

SourcePathType = Annotated[
    str | None,
    typer.Option(
        help="Path to an image file, a directory containing images or a video file for inference.",
    ),
]


@app.command()
def train(
    config: ConfigType = None,
    resume: Annotated[
        str | None,
        typer.Option(help="Resume training from this checkpoint."),
    ] = None,
    opts: OptsType = None,
):
    """Start training."""
    from luxonis_train.core import LuxonisModel

    LuxonisModel(config, opts).train(resume_weights=resume)


@app.command()
def test(
    config: ConfigType = None,
    view: ViewType = _ViewType.VAL,
    weights: WeightsType = None,
    opts: OptsType = None,
):
    """Evaluate model."""
    from luxonis_train.core import LuxonisModel

    LuxonisModel(config, opts).test(view=view.value, weights=weights)


@app.command()
def tune(config: ConfigType = None, opts: OptsType = None):
    """Start hyperparameter tuning."""
    from luxonis_train.core import LuxonisModel

    LuxonisModel(config, opts).tune()


@app.command()
def export(
    config: ConfigType = None,
    save_path: Annotated[
        Path | None,
        typer.Option(help="Path where to save the exported model."),
    ] = None,
    weights: WeightsType = None,
    opts: OptsType = None,
):
    """Export model."""
    from luxonis_train.core import LuxonisModel

    LuxonisModel(config, opts).export(
        onnx_save_path=save_path, weights=weights
    )


@app.command()
def infer(
    config: ConfigType = None,
    view: ViewType = _ViewType.VAL,
    save_dir: SaveDirType = None,
    source_path: SourcePathType = None,
    weights: WeightsType = None,
    opts: OptsType = None,
):
    """Run inference."""
    from luxonis_train.core import LuxonisModel

    LuxonisModel(config, opts).infer(
        view=view.value,
        save_dir=save_dir,
        source_path=source_path,
        weights=weights,
    )


@app.command()
def inspect(
    config: ConfigType = None,
    view: Annotated[
        _ViewType,
        typer.Option(
            ...,
            "--view",
            "-v",
            help="Which split of the dataset to inspect.",
            case_sensitive=False,
        ),
    ] = "train",  # type: ignore
    size_multiplier: Annotated[
        float,
        typer.Option(
            ...,
            "--size-multiplier",
            "-s",
            help=(
                "Multiplier for the image size. "
                "By default the images are shown in their original size. "
                "Use this option to scale them."
            ),
            show_default=False,
        ),
    ] = 1.0,
    opts: OptsType = None,
):
    """Inspect the dataset.

    To close the window press 'q' or 'Esc'.
    """
    import cv2
    from luxonis_ml.data import Augmentations, LabelType
    from luxonis_ml.data.utils.visualizations import visualize

    from luxonis_train.utils.registry import LOADERS

    cfg = Config.get_config(config, opts)
    train_augmentations = Augmentations(
        image_size=cfg.trainer.preprocessing.train_image_size,
        augmentations=[
            i.model_dump()
            for i in cfg.trainer.preprocessing.get_active_augmentations()
            if i.name != "Normalize"
        ],
        train_rgb=cfg.trainer.preprocessing.train_rgb,
        keep_aspect_ratio=cfg.trainer.preprocessing.keep_aspect_ratio,
    )
    val_augmentations = Augmentations(
        image_size=cfg.trainer.preprocessing.train_image_size,
        augmentations=[
            i.model_dump()
            for i in cfg.trainer.preprocessing.get_active_augmentations()
        ],
        train_rgb=cfg.trainer.preprocessing.train_rgb,
        keep_aspect_ratio=cfg.trainer.preprocessing.keep_aspect_ratio,
        only_normalize=True,
    )

    Loader = LOADERS.get(cfg.loader.name)
    loader = Loader(
        augmentations=(
            train_augmentations if view == "train" else val_augmentations
        ),
        view={
            "train": cfg.loader.train_view,
            "val": cfg.loader.val_view,
            "test": cfg.loader.test_view,
        }[view],
        image_source=cfg.loader.image_source,
        **cfg.loader.params,
    )

    for images, labels in loader:
        for img in images.values():
            if len(img.shape) != 3:
                raise ValueError(
                    "Only 3D images are supported for visualization."
                )
        np_images = {
            k: v.numpy().transpose(1, 2, 0) for k, v in images.items()
        }
        main_image = np_images[loader.image_source]
        main_image = cv2.cvtColor(main_image, cv2.COLOR_RGB2BGR)
        np_labels = {
            task: (label.numpy(), LabelType(task_type))
            for task, (label, task_type) in labels.items()
        }

        h, w, _ = main_image.shape
        new_h, new_w = int(h * size_multiplier), int(w * size_multiplier)
        main_image = cv2.resize(main_image, (new_w, new_h))
        viz = visualize(
            main_image,
            np_labels,
            loader.get_classes(),
        )
        cv2.imshow("Visualization", viz)
        if cv2.waitKey(0) in [ord("q"), 27]:
            break
    cv2.destroyAllWindows()


@app.command()
def archive(
    config: ConfigType = None,
    executable: Annotated[
        str | None,
        typer.Option(
            help="Path to the model file.", show_default=False, metavar="FILE"
        ),
    ] = None,
    weights: WeightsType = None,
    opts: OptsType = None,
):
    """Generate NN archive."""
    from luxonis_train.core import LuxonisModel

    LuxonisModel(str(config), opts).archive(path=executable, weights=weights)


def version_callback(value: bool):
    if value:
        typer.echo(f"LuxonisTrain Version: {version('luxonis_train')}")
        raise typer.Exit()


@app.callback()
def common(
    _: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=version_callback,
            help="Show version and exit.",
        ),
    ] = False,
    source: Annotated[
        Path | None,
        typer.Option(
            help="Path to a python file with custom components. "
            "Will be sourced before running the command.",
            metavar="FILE",
        ),
    ] = None,
):
    if source:
        exec(source.read_text(), globals(), globals())


if __name__ == "__main__":
    app()
