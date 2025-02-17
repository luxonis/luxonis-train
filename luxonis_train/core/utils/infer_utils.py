from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import torch
import torch.utils.data as torch_data
from luxonis_ml.data import DatasetIterator, LuxonisDataset
from torch import Tensor

import luxonis_train
from luxonis_train.attached_modules.visualizers import get_denormalized_images
from luxonis_train.loaders import LuxonisLoaderTorch
from luxonis_train.models.luxonis_output import LuxonisOutput

IMAGE_FORMATS = {
    ".bmp",
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
    ".dng",
    ".webp",
    ".mpo",
    ".pfm",
}
VIDEO_FORMATS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def process_visualizations(
    visualizations: dict[str, dict[str, Tensor]], batch_size: int
) -> dict[tuple[str, str], list[np.ndarray]]:
    """Render or save visualizations."""
    renders = defaultdict(list)

    for i in range(batch_size):
        for node_name, vzs in visualizations.items():
            for viz_name, viz_batch in vzs.items():
                viz = viz_batch[i]
                viz_arr = viz.detach().cpu().numpy().transpose(1, 2, 0)
                viz_arr = cv2.cvtColor(viz_arr, cv2.COLOR_RGB2BGR)
                renders[(node_name, viz_name)].append(viz_arr)

    return renders


def prepare_and_infer_image(
    model: "luxonis_train.core.LuxonisModel", img: Tensor
) -> LuxonisOutput:
    """Prepares the image for inference and runs the model."""
    img = model.loaders["val"].augment_test_image(img)  # type: ignore

    inputs = {
        "image": torch.tensor(img).unsqueeze(0).permute(0, 3, 1, 2).float()
    }
    images = get_denormalized_images(model.cfg, inputs)

    outputs = model.lightning_module.forward(
        inputs, images=images, compute_visualizations=True
    )
    return outputs


def window_closed() -> bool:  # pragma: no cover
    return cv2.waitKey(0) in {27, ord("q")}


def infer_from_video(
    model: "luxonis_train.core.LuxonisModel",
    video_path: str | Path,
    save_dir: Path | None,
) -> None:
    """Runs inference on individual frames from a video.

    @type model: L{LuxonisModel}
    @param model: The model to use for inference.
    @type video_path: str | Path
    @param video_path: The path to the video.
    @type save_dir: Path | None
    @param save_dir: The directory to save the visualizations to.
    @type show: bool
    @param show: Whether to display the visualizations.
    """

    cap = cv2.VideoCapture(filename=str(video_path))  # type: ignore

    writers: dict[str, cv2.VideoWriter] = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:  # pragma: no cover
            break
        if model.cfg.trainer.preprocessing.color_space == "RGB":
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # TODO: batched inference
        outputs = prepare_and_infer_image(model, torch.tensor(frame))
        renders = process_visualizations(outputs.visualizations, batch_size=1)

        for (node_name, viz_name), [viz] in renders.items():
            if save_dir is not None:
                name = f"{node_name}_{viz_name}"
                if name not in writers:
                    w, h = viz.shape[1], viz.shape[0]
                    writers[name] = cv2.VideoWriter(
                        filename=str(save_dir / f"{name}.mp4"),  # type: ignore
                        fourcc=cv2.VideoWriter_fourcc(*"mp4v"),  # type: ignore
                        fps=cap.get(cv2.CAP_PROP_FPS),  # type: ignore
                        frameSize=(w, h),  # type: ignore
                    )
                if name in writers:
                    writers[name].write(viz)
            else:  # pragma: no cover
                cv2.imshow(f"{node_name}/{viz_name}", viz)

        if not save_dir and window_closed():  # pragma: no cover
            break

    cap.release()
    cv2.destroyAllWindows()

    for writer in writers.values():
        writer.release()


def infer_from_loader(
    model: "luxonis_train.core.LuxonisModel",
    loader: torch_data.DataLoader,
    save_dir: Path | None,
    img_paths: list[Path] | None = None,
) -> None:
    """Runs inference on images from the dataset.

    @type model: L{LuxonisModel}
    @param model: The model to use for inference.
    @type loader: torch_data.DataLoader
    @param loader: The loader to use for inference.
    @type save_dir: str | Path | None
    @param save_dir: The directory to save the visualizations to.
    @type img_paths: list[Path] | None
    @param img_paths: The paths to the images.
    """

    predictions = model.pl_trainer.predict(model.lightning_module, loader)

    broken = False
    if predictions is None:
        return

    for i, outputs in enumerate(predictions):
        if broken:  # pragma: no cover
            break
        visualizations = outputs.visualizations  # type: ignore
        batch_size = next(
            iter(next(iter(visualizations.values())).values())
        ).shape[0]
        renders = process_visualizations(
            visualizations,
            batch_size=batch_size,
        )
        for j in range(batch_size):
            for (node_name, viz_name), visualizations in renders.items():
                viz = visualizations[j]
                if save_dir is not None:
                    if img_paths is not None:
                        img_path = img_paths[i * batch_size + j]
                        name = f"{img_path.stem}_{node_name}_{viz_name}"
                    else:
                        name = f"{node_name}_{viz_name}_{i * batch_size + j}"
                    name = name.replace("/", "-")
                    cv2.imwrite(str(save_dir / f"{name}.png"), viz)
                else:
                    cv2.imshow(f"{node_name}/{viz_name}", viz)

            if not save_dir and window_closed():  # pragma: no cover
                broken = True
                break

    cv2.destroyAllWindows()


def infer_from_directory(
    model: "luxonis_train.core.LuxonisModel",
    img_paths: Iterable[Path],
    save_dir: Path | None,
) -> None:
    """Runs inference on individual images from a directory.

    @type model: L{LuxonisModel}
    @param model: The model to use for inference.
    @type img_paths: Iterable[Path]
    @param img_paths: Iterable of paths to the images.
    @type save_dir: Path | None
    @param save_dir: The directory to save the visualizations to.
    """
    img_paths = list(img_paths)

    def generator() -> DatasetIterator:
        for img_path in img_paths:
            yield {
                "file": img_path,
            }

    dataset_name = "infer_from_directory"
    dataset = LuxonisDataset(dataset_name=dataset_name, delete_existing=True)
    dataset.add(generator())
    dataset.make_splits(
        {"train": 0.0, "val": 0.0, "test": 1.0}, replace_old_splits=True
    )

    loader = LuxonisLoaderTorch(
        dataset_name=dataset_name,
        view="test",
        height=model.cfg_preprocessing.train_image_size.height,
        width=model.cfg_preprocessing.train_image_size.width,
        augmentation_config=model.cfg_preprocessing.get_active_augmentations(),
        color_space=model.cfg_preprocessing.color_space,
        keep_aspect_ratio=model.cfg_preprocessing.keep_aspect_ratio,
    )
    loader = torch_data.DataLoader(
        loader, batch_size=model.cfg.trainer.batch_size, pin_memory=True
    )

    infer_from_loader(model, loader, save_dir, img_paths)

    dataset.delete_dataset()


def infer_from_dataset(
    model: "luxonis_train.core.LuxonisModel",
    view: Literal["train", "val", "test"],
    save_dir: Path | None,
) -> None:
    """Runs inference on images from the dataset.

    @type model: L{LuxonisModel}
    @param model: The model to use for inference.
    @type view: Literal["train", "val", "test"]
    @param view: The view of the dataset to use.
    @type save_dir: str | Path | None
    @param save_dir: The directory to save the visualizations to.
    """

    loader = model.pytorch_loaders[view]
    infer_from_loader(model, loader, save_dir)
