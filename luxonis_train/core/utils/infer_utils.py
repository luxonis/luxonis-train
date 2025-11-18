from collections import defaultdict
from collections.abc import Iterable
from contextlib import suppress
from pathlib import Path
from typing import Any, Literal, cast

import cv2
import numpy as np
import torch
import torch.utils.data as torch_data
from luxonis_ml.data import DatasetIterator, LuxonisDataset
from luxonis_ml.typing import PathType
from torch import Tensor
from torch.utils.data._utils.collate import default_collate

import luxonis_train as lxt
from luxonis_train.attached_modules.visualizers import get_denormalized_images
from luxonis_train.lightning import LuxonisOutput
from luxonis_train.loaders import LuxonisLoaderTorch
from luxonis_train.utils import Counter

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
    visualizations: dict[str, dict[str, Tensor]],
) -> dict[tuple[str, str], list[np.ndarray]]:
    """Render or save visualizations."""
    renders = defaultdict(list)

    for node_name, vzs in visualizations.items():
        for name, batch in vzs.items():
            for viz in batch:
                arr = viz.detach().cpu().numpy().transpose(1, 2, 0)
                arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                renders[(node_name, name)].append(arr)

    return renders


def prepare_and_infer_image(
    model: "lxt.LuxonisModel", images: dict[str, Tensor]
) -> LuxonisOutput:
    """Prepares the image for inference and runs the model."""
    npy_img = model.loaders["val"].augment_test_image(images)
    torch_img = torch.tensor(npy_img).unsqueeze(0).permute(0, 3, 1, 2).float()

    return model.lightning_module.forward(
        {"image": torch_img},
        images=get_denormalized_images(model.cfg, torch_img),
        compute_visualizations=True,
    )


def window_closed() -> bool:  # pragma: no cover
    return cv2.waitKey(0) in {27, ord("q")}


def infer_from_video(
    model: "lxt.LuxonisModel", video_path: PathType, save_dir: Path | None
) -> None:
    """Runs inference on individual frames from a video.

    @type model: L{LuxonisModel}
    @param model: The model to use for inference.
    @type video_path: PathType
    @param video_path: The path to the video.
    @type save_dir: Path | None
    @param save_dir: The directory to save the visualizations to.
    @type show: bool
    @param show: Whether to display the visualizations.
    """
    cap = cv2.VideoCapture(filename=str(video_path))

    writers: dict[str, cv2.VideoWriter] = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:  # pragma: no cover
            break
        if model.cfg.trainer.preprocessing.color_space == "RGB":
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # TODO: batched inference
        outputs = prepare_and_infer_image(
            model, {"image": torch.tensor(frame)}
        )
        renders = process_visualizations(outputs.visualizations)

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
    if save_dir is None:  # pragma: no cover
        with suppress(cv2.error):  # type: ignore
            cv2.destroyAllWindows()

    for writer in writers.values():
        writer.release()


def infer_from_loader(
    model: "lxt.LuxonisModel",
    loader: torch_data.DataLoader,
    save_dir: PathType | None,
    img_paths: list[PathType] | None = None,
) -> None:
    """Runs inference on images from the dataset.

    @type model: L{LuxonisModel}
    @param model: The model to use for inference.
    @type loader: torch_data.DataLoader
    @param loader: The loader to use for inference.
    @type save_dir: PathType | None
    @param save_dir: The directory to save the visualizations to.
    @type img_paths: list[Path] | None
    @param img_paths: The paths to the images.
    """
    predictions = model.pl_trainer.predict(model.lightning_module, loader)

    broken = False
    if predictions is None:  # pragma: no cover
        return

    counter = Counter()

    for outputs in predictions:
        if broken:  # pragma: no cover
            break
        assert isinstance(outputs, LuxonisOutput)
        visualizations = outputs.visualizations
        renders = process_visualizations(visualizations)
        batch_size = len(next(iter(renders.values())))
        for i in range(batch_size):
            if img_paths is not None:
                idx = counter()
            for (node_name, viz_name), visualizations in renders.items():
                viz = visualizations[i]
                if save_dir is not None:
                    save_dir = Path(save_dir)
                    if img_paths is not None:
                        img_path = Path(img_paths[idx])
                        name = f"{img_path.stem}_{node_name}_{viz_name}"
                    else:
                        name = f"{node_name}_{viz_name}_{counter()}"
                    name = name.replace("/", "-")
                    save_path = save_dir / f"{name}.png"
                    cv2.imwrite(str(save_path), viz)
                else:
                    cv2.imshow(f"{node_name}/{viz_name}", viz)

            if not save_dir and window_closed():  # pragma: no cover
                broken = True
                break

    if save_dir is None:  # pragma: no cover
        with suppress(cv2.error):  # type: ignore
            cv2.destroyAllWindows()


def create_loader_from_directory(
    img_paths: Iterable[PathType],
    model: "lxt.LuxonisModel",
    add_path_annotation: bool = False,
    batch_size: int | None = None,
) -> torch_data.DataLoader:
    """Creates a DataLoader from a directory of images.

    @type img_paths: Iterable[PathType]
    @param img_paths: Iterable of paths to the images.
    @type model: L{LuxonisModel}
    @param model: The model to use for inference.
    @type add_path_annotation: bool
    @param add_path_annotation: Whether to add the image path as an
        annotation in the dataset.
    @type batch_size: int | None
    @param batch_size: The batch size for the DataLoader. If None, uses
        the model's default batch size.
    @rtype: torch_data.DataLoader
    @return: The DataLoader for the images.
    """
    dataset_name = "infer_from_directory"
    dataset = LuxonisDataset(dataset_name=dataset_name, delete_local=True)

    def generator() -> DatasetIterator:
        for img_path in img_paths:
            data: dict[str, Any] = {"file": img_path}
            if add_path_annotation:
                data["annotation"] = {"metadata": {"path": str(img_path)}}
            yield data

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

    def collate_fix_paths(batch: list[tuple[Tensor, dict[str, Any]]]) -> Any:
        for _, m in batch:
            m["/metadata/path"] = (
                m["/metadata/path"]
                .view(-1)
                .byte()
                .cpu()
                .numpy()
                .tobytes()
                .rstrip(b"\0")
                .decode("utf-8", "ignore")
            )
        return default_collate(batch)

    return torch_data.DataLoader(
        loader,
        collate_fn=collate_fix_paths
        if add_path_annotation
        else default_collate,
        batch_size=batch_size or model.cfg.trainer.batch_size,
        pin_memory=True,
        shuffle=False,
    )


def infer_from_directory(
    model: "lxt.LuxonisModel",
    img_paths: Iterable[PathType],
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

    loader = create_loader_from_directory(img_paths, model)

    infer_from_loader(model, loader, save_dir, img_paths)
    inner_loader = cast(LuxonisLoaderTorch, loader.dataset)

    inner_loader.dataset.delete_dataset(delete_local=True)


def infer_from_dataset(
    model: "lxt.LuxonisModel",
    view: Literal["train", "val", "test"],
    save_dir: PathType | None,
) -> None:
    """Runs inference on images from the dataset.

    @type model: L{LuxonisModel}
    @param model: The model to use for inference.
    @type view: Literal["train", "val", "test"]
    @param view: The view of the dataset to use.
    @type save_dir: PathType | None
    @param save_dir: The directory to save the visualizations to.
    """
    loader = model.pytorch_loaders[view]
    infer_from_loader(model, loader, save_dir)
