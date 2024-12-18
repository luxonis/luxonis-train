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
    import os
    import zipfile
    from pathlib import Path

    import requests
    import torch.distributed as dist  # For rank checking in distributed environments
    from tqdm import tqdm

    if source:
        exec(source.read_text(), globals(), globals())

    def is_rank_0():
        """Check if the current process is rank 0 in a distributed
        environment."""
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank() == 0
        return True  # If not distributed, assume single-node, single-process (rank 0)

    # Paths for COCO dataset download and extraction
    coco_base_url = "http://images.cocodataset.org"
    coco_train_images_url = f"{coco_base_url}/zips/train2017.zip"
    coco_train_annotations_url = (
        f"{coco_base_url}/annotations/annotations_trainval2017.zip"
    )
    download_dir = Path("coco_dataset")
    train_images_dir = download_dir / "train2017"
    annotations_dir = download_dir / "annotations"

    def download_file(url, output_path):
        """Download a file from a URL with progress."""
        with requests.get(url, stream=True) as response:
            total = int(response.headers.get("content-length", 0))
            with (
                open(output_path, "wb") as file,
                tqdm(
                    desc=f"Downloading {output_path.name}",
                    total=total,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                ) as bar,
            ):
                for data in response.iter_content(chunk_size=1024):
                    file.write(data)
                    bar.update(len(data))

    def extract_zip(zip_path, extract_to):
        """Extract a zip file to a directory."""
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)

    # Only rank 0 downloads and extracts the dataset
    if is_rank_0():
        download_dir.mkdir(exist_ok=True)
        train_images_zip = download_dir / "train2017.zip"
        annotations_zip = download_dir / "annotations_trainval2017.zip"

        if not train_images_dir.exists():
            if not train_images_zip.exists():
                download_file(coco_train_images_url, train_images_zip)
            extract_zip(train_images_zip, download_dir)

        if not annotations_dir.exists():
            if not annotations_zip.exists():
                download_file(coco_train_annotations_url, annotations_zip)
            extract_zip(annotations_zip, download_dir)

        # Load the dataset
        import glob
        import json

        import cv2
        import numpy as np
        from luxonis_ml.data.datasets import LuxonisDataset
        from luxonis_ml.data.utils.enums import BucketStorage

        # Global lists for tracking yielded images
        yielded_train_images = []

        def COCO_generator():
            # Define the datasets for train
            datasets = [
                {
                    "annot_file": str(
                        annotations_dir / "instances_train2017.json"
                    ),
                    "img_dir": str(train_images_dir),
                    "split": "train",
                },
            ]

            for dataset in datasets:
                annot_file = dataset["annot_file"]
                img_dir = dataset["img_dir"]
                split = dataset["split"]

                # Get paths to images sorted by number
                im_paths = glob.glob(os.path.join(img_dir, "*.jpg"))
                nums = np.array(
                    [
                        int(os.path.splitext(os.path.basename(path))[0])
                        for path in im_paths
                    ]
                )
                idxs = np.argsort(nums)
                im_paths = list(np.array(im_paths)[idxs])

                # Load annotations
                with open(annot_file) as file:
                    data = json.load(file)
                imgs = data["images"]
                anns = data["annotations"]
                cats = data["categories"]

                # Create dictionaries for quick lookups
                img_dict = {img["file_name"]: img for img in imgs}
                ann_dict = {}
                for ann in anns:
                    img_id = ann["image_id"]
                    if img_id not in ann_dict:
                        ann_dict[img_id] = []
                    ann_dict[img_id].append(ann)

                # Process each image and its annotations
                for path in tqdm(im_paths):
                    gran = os.path.basename(path)
                    img = img_dict.get(gran, None)
                    if img is None:
                        continue
                    img_id = img["id"]
                    img_anns = ann_dict.get(img_id, [])

                    if not img_anns:
                        continue

                    im = cv2.imread(path)
                    if im is None:
                        continue

                    height, width, _ = im.shape

                    for _, ann in enumerate(img_anns):
                        if ann.get("iscrowd", True):
                            continue

                        cls = [
                            cat
                            for cat in cats
                            if cat["id"] == ann["category_id"]
                        ][0]["name"]

                        x, y, w, h = ann["bbox"]

                        if split == "train":
                            if path not in yielded_train_images:
                                yielded_train_images.append(path)

                        yield {
                            "file": path,
                            "annotation": {
                                "type": "boundingbox",
                                "class": cls,
                                "x": x / width,
                                "y": y / height,
                                "w": w / width,
                                "h": h / height,
                            },
                        }

        dataset_name = "coco_local_train"
        bucket_storage = BucketStorage.LOCAL
        dataset = LuxonisDataset(
            dataset_name,
            bucket_storage=bucket_storage,
            delete_existing=True,
            delete_remote=True,
        )
        definitions = {
            "train": yielded_train_images,
        }
        dataset.add(COCO_generator(), batch_size=100_000_000)
        dataset.make_splits(definitions=definitions)

    # Synchronize across all ranks after dataset preparation
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


if __name__ == "__main__":
    app()
