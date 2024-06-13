import os
from enum import Enum
from importlib.metadata import version
from pathlib import Path
from typing import Annotated, Optional

import cv2
import typer
from torch.utils.data import DataLoader

from luxonis_train.utils.registry import LOADERS

app = typer.Typer(
    help="Luxonis Train CLI",
    add_completion=False,
    pretty_exceptions_show_locals=False,
)


class View(str, Enum):
    train = "train"
    val = "val"
    test = "test"

    def __str__(self):
        return self.value


ConfigType = Annotated[
    Optional[Path],
    typer.Option(
        help="Path to the configuration file.",
        show_default=False,
    ),
]

OptsType = Annotated[
    Optional[list[str]],
    typer.Argument(
        help="A list of optional CLI overrides of the config file.",
        show_default=False,
    ),
]

ViewType = Annotated[View, typer.Option(help="Which dataset view to use.")]

SaveDirType = Annotated[
    Optional[Path],
    typer.Option(help="Where to save the inference results."),
]


@app.command()
def train(
    config: ConfigType = None,
    resume: Annotated[
        Optional[str], typer.Option(help="Resume training from this checkpoint.")
    ] = None,
    opts: OptsType = None,
):
    """Start training."""
    from luxonis_train.core import Trainer

    Trainer(str(config), opts, resume=resume).train()


@app.command()
def eval(config: ConfigType = None, view: ViewType = View.val, opts: OptsType = None):
    """Evaluate model."""
    from luxonis_train.core import Trainer

    Trainer(str(config), opts).test(view=view.name)


@app.command()
def tune(config: ConfigType = None, opts: OptsType = None):
    """Start hyperparameter tuning."""
    from luxonis_train.core import Tuner

    Tuner(str(config), opts).tune()


@app.command()
def export(config: ConfigType = None, opts: OptsType = None):
    """Export model."""
    from luxonis_train.core import Exporter

    Exporter(str(config), opts).export()


@app.command()
def infer(
    config: ConfigType = None,
    view: ViewType = View.val,
    save_dir: SaveDirType = None,
    opts: OptsType = None,
):
    """Run inference."""
    from luxonis_train.core import Inferer

    Inferer(str(config), opts, view=view.name, save_dir=save_dir).infer()


@app.command()
def inspect(
    config: ConfigType = None,
    view: ViewType = View.val,
    save_dir: SaveDirType = None,
    opts: OptsType = None,
):
    """Inspect dataset."""
    from lightning.pytorch import seed_everything
    from luxonis_ml.data import Augmentations

    from luxonis_train.attached_modules.visualizers.utils import (
        draw_bounding_box_labels,
        draw_keypoint_labels,
        draw_segmentation_labels,
        get_unnormalized_images,
    )
    from luxonis_train.utils.config import Config
    from luxonis_train.utils.loaders import collate_fn
    from luxonis_train.utils.types import LabelType

    overrides = {}
    if opts:
        if len(opts) % 2 != 0:
            raise ValueError("Override options should be a list of key-value pairs")

        for i in range(0, len(opts), 2):
            overrides[opts[i]] = opts[i + 1]

    cfg = Config.get_config(str(config), overrides)
    if cfg.trainer.seed is not None:
        seed_everything(cfg.trainer.seed, workers=True)

    image_size = cfg.trainer.preprocessing.train_image_size

    augmentations = Augmentations(
        image_size=image_size,
        augmentations=[
            i.model_dump() for i in cfg.trainer.preprocessing.get_active_augmentations()
        ],
        train_rgb=cfg.trainer.preprocessing.train_rgb,
        keep_aspect_ratio=cfg.trainer.preprocessing.keep_aspect_ratio,
        only_normalize=view != "train",
    )

    loader = LOADERS.get(cfg.loader.name)(
        view=view, augmentations=augmentations, **cfg.loader.params
    )

    pytorch_loader = DataLoader(
        loader,
        batch_size=1,
        num_workers=0,
        collate_fn=collate_fn,
    )

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    counter = 0
    for data in pytorch_loader:
        imgs, task_dict = data
        for task, label_dict in task_dict.items():
            images = get_unnormalized_images(cfg, imgs)
            for i, img in enumerate(images):
                for label_type, labels in label_dict.items():
                    if label_type == LabelType.CLASSIFICATION:
                        continue
                    elif label_type == LabelType.BOUNDINGBOX:
                        img = draw_bounding_box_labels(
                            img,
                            labels[labels[:, 0] == i][:, 2:],
                            colors="yellow",
                            width=1,
                        )
                    elif label_type == LabelType.KEYPOINTS:
                        img = draw_keypoint_labels(
                            img, labels[labels[:, 0] == i][:, 1:], colors="red"
                        )
                    elif label_type == LabelType.SEGMENTATION:
                        img = draw_segmentation_labels(
                            img, labels[i], alpha=0.8, colors="#5050FF"
                        )

                img_arr = img.permute(1, 2, 0).numpy()
                img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
                if save_dir is not None:
                    counter += 1
                    cv2.imwrite(
                        os.path.join(save_dir, f"{counter}_{task}.png"), img_arr
                    )
                else:
                    cv2.imshow(task, img_arr)
        if save_dir is None and cv2.waitKey() == ord("q"):
            exit()


@app.command()
def archive(
    executable: Annotated[
        Optional[Path], typer.Option(help="Path to the model file.", show_default=False)
    ],
    config: ConfigType = None,
    opts: OptsType = None,
):
    """Generate NN archive."""
    from luxonis_train.core import Archiver

    Archiver(str(config), opts).archive(executable)


def version_callback(value: bool):
    if value:
        typer.echo(f"LuxonisTrain Version: {version(__package__)}")
        raise typer.Exit()


@app.callback()
def common(
    _: Annotated[
        bool,
        typer.Option(
            "--version", callback=version_callback, help="Show version and exit."
        ),
    ] = False,
    source: Annotated[
        Optional[Path],
        typer.Option(
            help="Path to a python file with custom components. "
            "Will be sourced before running the command.",
            metavar="FILE",
        ),
    ] = None,
):
    if source:
        exec(source.read_text())


if __name__ == "__main__":
    app()
