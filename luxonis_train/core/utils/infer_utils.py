from pathlib import Path

import cv2
import torch
from torch import Tensor

from luxonis_train.attached_modules.visualizers import get_unnormalized_images
from luxonis_train.enums import TaskType


def render_visualizations(
    visualizations: dict[str, dict[str, Tensor]], save_dir: str | Path | None
) -> None:
    """Render or save visualizations."""
    save_dir = Path(save_dir) if save_dir is not None else None
    if save_dir is not None:
        save_dir.mkdir(exist_ok=True, parents=True)

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
                else:
                    cv2.imshow(name, viz_arr)

    if save_dir is None:
        if cv2.waitKey(0) == ord("q"):
            exit()


def process_single_image(
    model, img_path: Path, view: str, save_dir: str | Path | None
) -> None:
    """Handles the inference on a single image."""
    img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
    img, _ = model.val_augmentations([(img, {})])
    labels = create_dummy_labels(model, view, img.shape)
    inputs = {
        "image": torch.tensor(img).unsqueeze(0).permute(0, 3, 1, 2).float()
    }
    images = get_unnormalized_images(model.cfg, inputs)

    outputs = model.lightning_module.forward(
        inputs, labels, images=images, compute_visualizations=True
    )
    render_visualizations(outputs.visualizations, save_dir)


def process_directory_images(
    model, dir_path: Path, view: str, save_dir: str | Path | None
) -> None:
    """Handles inference for multiple images in a directory."""
    image_files = [
        f
        for f in dir_path.iterdir()
        if f.suffix.lower() in {".png", ".jpg", ".jpeg"}
    ]
    for image_file in image_files:
        process_single_image(model, image_file, view, save_dir)


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
    nk = model.loaders[view].get_n_keypoints()["keypoints"]

    for task in tasks:
        if task == "classification":
            labels[task] = [-1, TaskType.CLASSIFICATION]
        elif task == "keypoints":
            labels[task] = [torch.zeros((1, nk * 3 + 2)), TaskType.KEYPOINTS]
        elif task == "segmentation":
            labels[task] = [torch.zeros((1, h, w)), TaskType.SEGMENTATION]
        elif task == "boundingbox":
            labels[task] = [
                torch.tensor([[-1, 0, 0, 0, 0, 0]]),
                TaskType.BOUNDINGBOX,
            ]

    return labels
