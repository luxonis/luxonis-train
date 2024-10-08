from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import torch
from torch import Tensor

import luxonis_train
from luxonis_train.attached_modules.visualizers import get_unnormalized_images

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
    model: "luxonis_train.core.LuxonisModel",
    img: np.ndarray,
):
    """Prepares the image for inference and runs the model."""
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img, _ = model.val_augmentations([(img, {})])

    inputs = {
        "image": torch.tensor(img).unsqueeze(0).permute(0, 3, 1, 2).float()
    }
    images = get_unnormalized_images(model.cfg, inputs)

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

        # TODO: batched inference
        outputs = prepare_and_infer_image(model, frame)
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
    for img_path in img_paths:
        img = cv2.imread(str(img_path))
        outputs = prepare_and_infer_image(model, img)
        renders = process_visualizations(outputs.visualizations, batch_size=1)

        for (node_name, viz_name), [viz] in renders.items():
            if save_dir is not None:
                cv2.imwrite(
                    str(
                        save_dir
                        / f"{img_path.stem}_{node_name}_{viz_name}.png"
                    ),
                    viz,
                )
            else:  # pragma: no cover
                cv2.imshow(f"{node_name}/{viz_name}", viz)

        if not save_dir and window_closed():  # pragma: no cover
            break

    cv2.destroyAllWindows()


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
    broken = False
    for i, (inputs, labels) in enumerate(model.pytorch_loaders[view]):
        if broken:  # pragma: no cover
            break

        images = get_unnormalized_images(model.cfg, inputs)
        batch_size = images.shape[0]
        outputs = model.lightning_module.forward(
            inputs, labels, images=images, compute_visualizations=True
        )
        renders = process_visualizations(
            outputs.visualizations,
            batch_size=batch_size,
        )
        for j in range(batch_size):
            for (node_name, viz_name), visualizations in renders.items():
                viz = visualizations[j]
                if save_dir is not None:
                    name = f"{node_name}_{viz_name}"
                    cv2.imwrite(
                        str(save_dir / f"{name}_{i * batch_size + j}.png"), viz
                    )
                else:
                    cv2.imshow(f"{node_name}/{viz_name}", viz)

            if not save_dir and window_closed():  # pragma: no cover
                broken = True
                break

    cv2.destroyAllWindows()
