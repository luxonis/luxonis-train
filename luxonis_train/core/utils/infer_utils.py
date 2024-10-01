from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import tqdm
from torch import Tensor

from luxonis_train.attached_modules.visualizers import get_unnormalized_images
from luxonis_train.enums import TaskType


def render_visualizations(
    visualizations: dict[str, dict[str, Tensor]],
    save_dir: str | Path | None,
    show: bool = True,
) -> dict[str, list[np.ndarray]]:
    """Render or save visualizations."""
    save_dir = Path(save_dir) if save_dir is not None else None
    if save_dir is not None:
        save_dir.mkdir(exist_ok=True, parents=True)

    rendered_visualizations = defaultdict(list)
    i = 0
    for node_name, vzs in visualizations.items():
        for viz_name, viz_batch in vzs.items():
            for i, viz in enumerate(viz_batch):
                viz_arr = viz.detach().cpu().numpy().transpose(1, 2, 0)
                viz_arr = cv2.cvtColor(viz_arr, cv2.COLOR_RGB2BGR)
                name = f"{node_name}/{viz_name}/{i}"
                if save_dir is not None:
                    name = name.replace("/", "_")
                    cv2.imwrite(str(save_dir / f"{name}_{i}.png"), viz_arr)
                    i += 1
                elif show:
                    cv2.imshow(name, viz_arr)
                else:
                    rendered_visualizations[name].append(viz_arr)

    if save_dir is None and show:
        if cv2.waitKey(0) == ord("q"):
            exit()

    return rendered_visualizations


def prepare_and_infer_image(model, img: np.ndarray, labels: dict, view: str):
    """Prepares the image for inference and runs the model."""
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img, _ = (
        model.train_augmentations([(img, {})])
        if view == "train"
        else model.val_augmentations([(img, {})])
    )

    inputs = {
        "image": torch.tensor(img).unsqueeze(0).permute(0, 3, 1, 2).float()
    }
    images = get_unnormalized_images(model.cfg, inputs)

    outputs = model.lightning_module.forward(
        inputs, labels, images=images, compute_visualizations=True
    )
    return outputs


def process_video(
    model,
    video_path: str | Path,
    view: str,
    save_dir: str | Path | None,
    show: bool = False,
) -> None:
    """Handles inference on a video."""
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = tqdm.tqdm(
        total=total_frames, position=0, leave=True, desc="Processing video"
    )

    if save_dir is not None:
        out_writers = {}
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)

    labels = create_dummy_labels(
        model, view, (int(cap.get(4)), int(cap.get(3)), 3)
    )

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        outputs = prepare_and_infer_image(model, frame, labels, view)
        rendered_visualizations = render_visualizations(
            outputs.visualizations, None, show
        )
        if save_dir is not None:
            for name, viz_arrs in rendered_visualizations.items():
                if name not in out_writers:
                    out_writers[name] = cv2.VideoWriter(
                        str(save_dir / f"{name.replace('/', '-')}.mp4"),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        cap.get(cv2.CAP_PROP_FPS),
                        (viz_arrs[0].shape[1], viz_arrs[0].shape[0]),
                    )
                for viz_arr in viz_arrs:
                    out_writers[name].write(viz_arr)

        progress_bar.update(1)

    if save_dir is not None:
        for writer in out_writers.values():
            writer.release()

    cap.release()
    progress_bar.close()


def process_images(
    model, img_paths: list[Path], view: str, save_dir: str | Path | None
) -> None:
    """Handles inference on one or more images."""
    first_image = cv2.cvtColor(
        cv2.imread(str(img_paths[0])), cv2.COLOR_BGR2RGB
    )
    labels = create_dummy_labels(model, view, first_image.shape)
    for img_path in img_paths:
        img = cv2.imread(str(img_path))
        outputs = prepare_and_infer_image(model, img, labels, view)
        render_visualizations(outputs.visualizations, save_dir)


def process_dataset_images(
    model, view: str, save_dir: str | Path | None
) -> None:
    """Handles the inference on dataset images."""
    for inputs, labels in model.pytorch_loaders[view]:
        images = get_unnormalized_images(model.cfg, inputs)
        outputs = model.lightning_module.forward(
            inputs, labels, images=images, compute_visualizations=True
        )
        render_visualizations(outputs.visualizations, save_dir)


def create_dummy_labels(model, view: str, img_shape: tuple) -> dict:
    """Prepares the labels for different tasks (classification,
    keypoints, etc.)."""
    tasks = list(model.loaders["train"].get_classes().keys())
    h, w, _ = img_shape
    labels = {}

    for task in tasks:
        if task == "classification":
            labels[task] = [-1, TaskType.CLASSIFICATION]
        elif task == "keypoints":
            nk = model.loaders[view].get_n_keypoints()["keypoints"]
            labels[task] = [torch.zeros((1, nk * 3 + 2)), TaskType.KEYPOINTS]
        elif task == "segmentation":
            labels[task] = [torch.zeros((1, h, w)), TaskType.SEGMENTATION]
        elif task == "boundingbox":
            labels[task] = [
                torch.tensor([[-1, 0, 0, 0, 0, 0]]),
                TaskType.BOUNDINGBOX,
            ]

    return labels
