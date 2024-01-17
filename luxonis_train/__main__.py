import os
from enum import Enum
from importlib.metadata import version
from pathlib import Path
from typing import Annotated, Optional

import cv2
import torch
import typer

app = typer.Typer(help="Luxonis Train CLI", add_completion=False)


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
def train(config: ConfigType = None, opts: OptsType = None):
    """Start training."""
    from luxonis_train.core import Trainer

    Trainer(str(config), opts).train()


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
    from luxonis_ml.data import (
        LuxonisDataset,
        TrainAugmentations,
        ValAugmentations,
    )

    from luxonis_train.attached_modules.visualizers.utils import (
        draw_bounding_box_labels,
        draw_keypoint_labels,
        draw_segmentation_labels,
        get_unnormalized_images,
    )
    from luxonis_train.utils.config import Config
    from luxonis_train.utils.loaders import LuxonisLoaderTorch, collate_fn
    from luxonis_train.utils.types import LabelType

    overrides = {}
    if opts:
        if len(opts) % 2 != 0:
            raise ValueError("Override options should be a list of key-value pairs")

        for i in range(0, len(opts), 2):
            overrides[opts[i]] = opts[i + 1]

    cfg = Config.get_config(str(config), overrides)

    image_size = cfg.trainer.preprocessing.train_image_size

    dataset = LuxonisDataset(
        dataset_name=cfg.dataset.name,
        team_id=cfg.dataset.team_id,
        dataset_id=cfg.dataset.id,
        bucket_type=cfg.dataset.bucket_type,
        bucket_storage=cfg.dataset.bucket_storage,
    )
    augmentations = (
        TrainAugmentations(
            image_size=image_size,
            augmentations=[
                i.model_dump() for i in cfg.trainer.preprocessing.augmentations
            ],
            train_rgb=cfg.trainer.preprocessing.train_rgb,
            keep_aspect_ratio=cfg.trainer.preprocessing.keep_aspect_ratio,
        )
        if view == "train"
        else ValAugmentations(
            image_size=image_size,
            augmentations=[
                i.model_dump() for i in cfg.trainer.preprocessing.augmentations
            ],
            train_rgb=cfg.trainer.preprocessing.train_rgb,
            keep_aspect_ratio=cfg.trainer.preprocessing.keep_aspect_ratio,
        )
    )

    loader_train = LuxonisLoaderTorch(
        dataset,
        view=view,
        augmentations=augmentations,
    )

    pytorch_loader_train = torch.utils.data.DataLoader(
        loader_train,
        batch_size=4,
        num_workers=1,
        collate_fn=collate_fn,
    )

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    counter = 0
    for data in pytorch_loader_train:
        imgs, label_dict = data
        images = get_unnormalized_images(cfg, imgs)
        for i, img in enumerate(images):
            for label_type, labels in label_dict.items():
                if label_type == LabelType.CLASSIFICATION:
                    continue
                elif label_type == LabelType.BOUNDINGBOX:
                    img = draw_bounding_box_labels(
                        img, labels[labels[:, 0] == i][:, 2:], colors="yellow", width=1
                    )
                elif label_type == LabelType.KEYPOINT:
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
                cv2.imwrite(os.path.join(save_dir, f"{counter}.png"), img_arr)
            else:
                cv2.imshow("img", img_arr)
                if cv2.waitKey() == ord("q"):
                    exit()


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
):
    ...


def main():
    app()


if __name__ == "__main__":
    main()
