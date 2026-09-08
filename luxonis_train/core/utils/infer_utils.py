from collections import defaultdict
from collections.abc import Generator, Iterable
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any, Literal, cast

import cv2
import numpy as np
import torch
import torch.utils.data as torch_data
from lightning.pytorch.callbacks import BasePredictionWriter
from loguru import logger
from luxonis_ml.data import LuxonisDataset
from luxonis_ml.typing import PathType
from torch import Tensor
from torch.utils.data._utils.collate import default_collate

import luxonis_train as lxt
from luxonis_train.attached_modules.visualizers import get_denormalized_images
from luxonis_train.lightning import LuxonisOutput
from luxonis_train.loaders import LuxonisLoaderTorch
from luxonis_train.typing import Labels
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
    """Prepare the image for inference and runs the model."""
    npy_img = model.loaders["val"].augment_test_image(images)
    torch_img = torch.tensor(npy_img).unsqueeze(0).permute(0, 3, 1, 2).float()

    return model.lightning_module.full_forward(
        {model.lightning_module.image_source: torch_img},
        images=get_denormalized_images(model.cfg, torch_img),
        compute_visualizations=True,
    )


def window_closed() -> bool:  # pragma: no cover
    return cv2.waitKey(0) in {27, ord("q")}


def infer_from_video(
    model: "lxt.LuxonisModel", video_path: PathType, save_dir: Path | None
) -> None:
    """Run inference on individual frames from a video.

    Args:
        model (LuxonisModel): Model to use for inference.
        video_path (`PathType <luxonis_ml.typing.PathType>`): ``Path`` to the video.
        save_dir (``Path | None``): Directory where visualizations are saved. If
            ``None``, visualizations are displayed on screen.

    """
    cap = cv2.VideoCapture(filename=str(video_path))

    writers: dict[str, cv2.VideoWriter] = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:  # pragma: no cover
            break
        renders = _infer_video_frame(model, frame)
        _write_or_show_renders(renders, writers, save_dir, cap)

        if not save_dir and window_closed():  # pragma: no cover
            break

    cap.release()
    _close_video_windows(save_dir)
    for writer in writers.values():
        writer.release()


def _infer_video_frame(
    model: "lxt.LuxonisModel", frame: np.ndarray
) -> dict[tuple[str, str], list[np.ndarray]]:
    if model.cfg.trainer.preprocessing.color_space == "RGB":
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # TODO: batched inference
    outputs = prepare_and_infer_image(model, {"image": torch.tensor(frame)})
    return process_visualizations(outputs.visualizations)


def _write_or_show_renders(
    renders: dict[tuple[str, str], list[np.ndarray]],
    writers: dict[str, cv2.VideoWriter],
    save_dir: Path | None,
    capture: cv2.VideoCapture,
) -> None:
    for (node_name, viz_name), [viz] in renders.items():
        if save_dir is None:  # pragma: no cover
            cv2.imshow(f"{node_name}/{viz_name}", viz)
            continue
        name = f"{node_name}_{viz_name}"
        writer = writers.get(name)
        if writer is None:
            writer = _create_video_writer(name, viz, save_dir, capture)
            writers[name] = writer
        writer.write(viz)


def _create_video_writer(
    name: str,
    visualization: np.ndarray,
    save_dir: Path,
    capture: cv2.VideoCapture,
) -> cv2.VideoWriter:
    width, height = visualization.shape[1], visualization.shape[0]
    return cv2.VideoWriter(
        filename=str(save_dir / f"{name}.mp4"),
        fourcc=cv2.VideoWriter.fourcc(*"mp4v"),
        fps=capture.get(cv2.CAP_PROP_FPS),
        frameSize=(width, height),
    )


def _close_video_windows(save_dir: Path | None) -> None:
    if save_dir is not None:
        return
    with suppress(cv2.error):  # type: ignore
        cv2.destroyAllWindows()


def infer_from_loader(
    model: "lxt.LuxonisModel",
    loader: torch_data.DataLoader,
    save_dir: PathType | None,
    img_paths: list[PathType] | None = None,
) -> None:
    """Run inference on images from the dataset.

    Args:
        model (LuxonisModel): Model to use for inference.
        loader (torch_data.DataLoader): Loader to use for inference.
        save_dir (``PathType | None``): Directory where visualizations are saved.
            If ``None``, visualizations are displayed on screen.
        img_paths (``list[PathType] | None``): Paths to the images.

    """
    if save_dir is not None:
        save_dir = Path(save_dir)
        writer = _VisualizationPredictionWriter(save_dir, img_paths)
        with _temporary_callback(model.pl_trainer, writer):
            model.pl_trainer.predict(
                model.lightning_module,
                loader,
                return_predictions=False,
            )
        return

    predictions = model.pl_trainer.predict(model.lightning_module, loader)

    broken = False
    if predictions is None:  # pragma: no cover
        return

    for outputs in predictions:
        if broken:  # pragma: no cover
            break
        assert isinstance(outputs, LuxonisOutput)
        visualizations = outputs.visualizations
        renders = process_visualizations(visualizations)
        batch_size = len(next(iter(renders.values())))
        for i in range(batch_size):
            for (node_name, viz_name), visualizations in renders.items():
                viz = visualizations[i]
                cv2.imshow(f"{node_name}/{viz_name}", viz)

            if window_closed():  # pragma: no cover
                broken = True
                break

    with suppress(cv2.error):  # pragma: no cover
        cv2.destroyAllWindows()


class _VisualizationPredictionWriter(BasePredictionWriter):
    def __init__(
        self,
        save_dir: Path,
        img_paths: list[PathType] | None = None,
    ) -> None:
        super().__init__(write_interval="batch")
        self.save_dir = save_dir
        self.img_paths = img_paths
        self.counter = Counter()

    def write_on_batch_end(
        self,
        trainer: Any,
        pl_module: Any,
        prediction: Any,
        batch_indices: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        del trainer, pl_module, batch_indices, batch, batch_idx, dataloader_idx

        assert isinstance(prediction, LuxonisOutput)
        renders = process_visualizations(prediction.visualizations)
        _save_renders_batch(
            renders,
            self.save_dir,
            self.counter,
            self.img_paths,
        )


@contextmanager
def _temporary_callback(
    trainer: Any, callback: Any
) -> Generator[None, None, None]:
    trainer.callbacks.append(callback)
    try:
        yield
    finally:
        with suppress(ValueError):
            trainer.callbacks.remove(callback)


def _save_renders_batch(
    renders: dict[tuple[str, str], list[np.ndarray]],
    save_dir: Path,
    counter: Counter,
    img_paths: list[PathType] | None = None,
) -> None:
    if not renders:
        return

    batch_size = len(next(iter(renders.values())))
    for i in range(batch_size):
        img_path: Path | None = None
        if img_paths is not None:
            img_path = Path(img_paths[counter()])
        for (node_name, viz_name), visualizations in renders.items():
            viz = visualizations[i]
            if img_path is not None:
                name = f"{img_path.stem}_{node_name}_{viz_name}"
            else:
                name = f"{node_name}_{viz_name}_{counter()}"
            name = name.replace("/", "-")
            cv2.imwrite(str(save_dir / f"{name}.png"), viz)


def create_loader_from_directory(
    img_paths: Iterable[PathType],
    model: "lxt.LuxonisModel",
    batch_size: int | None = None,
    return_sample_metadata: bool = False,
) -> torch_data.DataLoader:
    """Create a DataLoader from a directory of images.

    Args:
        img_paths (``Iterable[PathType]``): Iterable of paths to the images.
        model (LuxonisModel): Model to use for inference.
        batch_size (int | None): Batch size for the DataLoader. If ``None``,
            the model's default batch size is used.
        return_sample_metadata (bool): Whether the DataLoader also returns the
            per-sample metadata. Passed to ``LuxonisLoaderTorch``.

    Returns:
        torch_data.DataLoader: The DataLoader for the images.

    """
    dataset_name = "infer_from_directory"
    dataset = LuxonisDataset(dataset_name=dataset_name, delete_local=True)

    dataset.add(
        {
            "file": img_path,
            "sample_metadata": {"path": str(img_path)},
        }
        for img_path in img_paths
    )
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
        return_sample_metadata=return_sample_metadata,
    )

    def collate_with_metadata(
        batch: list[tuple[Any, Labels, dict[str, Any]]],
    ) -> Any:
        samples = [(img, labels) for img, labels, _ in batch]
        sample_metadata = [metadata for *_, metadata in batch]
        inputs, labels = default_collate(samples)
        return inputs, labels, sample_metadata

    return torch_data.DataLoader(
        loader,
        collate_fn=collate_with_metadata
        if return_sample_metadata
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
    """Run inference on individual images from a directory.

    Args:
        model (LuxonisModel): Model to use for inference.
        img_paths (``Iterable[PathType]``): Iterable of paths to the images.
        save_dir (``Path | None``): Directory where visualizations are saved. If
            ``None``, visualizations are displayed on screen.

    """
    img_paths = list(img_paths)

    loader = create_loader_from_directory(img_paths, model)

    infer_from_loader(model, loader, save_dir, img_paths)
    inner_loader = cast(LuxonisLoaderTorch, loader.dataset)

    inner_loader.dataset.delete_dataset(delete_local=True)


class _LimitedLoader:
    """Wrapper around a DataLoader that limits the number of batches."""

    def __init__(self, loader: torch_data.DataLoader, n_batches: int) -> None:
        self._loader = loader
        self._n_batches = n_batches

    def __iter__(self):
        for i, batch in enumerate(self._loader):
            if i >= self._n_batches:
                break
            yield batch

    def __len__(self) -> int:
        return min(self._n_batches, len(self._loader))


def infer_from_dataset(
    model: "lxt.LuxonisModel",
    view: Literal["train", "val", "test"],
    save_dir: PathType | None,
) -> None:
    """Run inference on images from the dataset.

    Args:
        model (LuxonisModel): Model to use for inference.
        view (``Literal["train", "val", "test"]``): Dataset view to use.
        save_dir (``PathType | None``): Directory where visualizations are saved.
            If ``None``, visualizations are displayed on screen.

    """
    loader = model.pytorch_loaders[view]
    overfit_batches = model.cfg.trainer.overfit_batches
    if overfit_batches > 0 and view == "train":
        logger.warning(
            f"Using limited loader with {overfit_batches} batches because "
            "`trainer.overfit_batches` is set. If this is not intended, "
            "remove `overfit_batches` from your config."
        )
        loader = _LimitedLoader(loader, overfit_batches)  # type: ignore[assignment]
    infer_from_loader(model, loader, save_dir)  # type: ignore[arg-type]
