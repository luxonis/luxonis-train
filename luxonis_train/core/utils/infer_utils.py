"""Inference over an image, a video, a directory, or a dataset view.

`LuxonisModel.infer` calls these helpers. They run the model, convert
the images of its visualizers to OpenCV arrays, and save the arrays or
show them in a window. `LuxonisModel.annotate` also reads its images
through `create_loader_from_directory`.

``VIDEO_FORMATS`` holds the lowercase file extensions that
`LuxonisModel.infer` reads as a video. ``IMAGE_FORMATS`` holds the
lowercase file extensions of the images that `LuxonisModel.infer` and
`LuxonisModel.annotate` select in a directory.
`LuxonisLoaderPerlinNoise` also selects its anomaly source images with
it.

"""

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
    """Convert the images of the visualizers into OpenCV arrays.

    The function takes the ``visualizations`` of a `LuxonisOutput` and
    turns every image of every batch into a NumPy array. It detaches
    each image, moves it to the CPU, and transposes it from
    ``[C, H, W]`` to ``[H, W, C]``. It then converts the channels from
    RGB to BGR with ``cv2.cvtColor``, because OpenCV writes and shows
    BGR. An image with four channels loses its fourth channel.

    Args:
        visualizations (``dict[str, dict[str, Tensor]]``): The image
            batch of every visualizer, of shape ``[B, C, H, W]``, keyed
            by node name and visualizer name.

    Returns:
        ``dict[tuple[str, str], list[np.ndarray]]``: The ``B`` images
        of every visualizer, each of shape ``[H, W, 3]`` in BGR, keyed
        by the pair of node name and visualizer name. The result is a
        ``defaultdict``, so a missing key gives an empty list.

    Example:
        >>> import torch
        >>> image = torch.zeros(2, 3, 4, 5, dtype=torch.uint8)
        >>> image[:, 0] = 255  # red in RGB
        >>> renders = process_visualizations({"head": {"boxes": image}})
        >>> sorted(renders)
        [('head', 'boxes')]
        >>> [render.shape for render in renders["head", "boxes"]]
        [(4, 5, 3), (4, 5, 3)]
        >>> renders["head", "boxes"][0][0, 0].tolist()
        [0, 0, 255]

    """
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
    """Preprocess one raw image with the ``val`` loader and run the
    model on it.

    The function passes ``images`` to ``augment_test_image`` of the
    ``val`` loader of ``model``. That method returns one image of shape
    ``[H, W, C]``. `LuxonisLoaderTorch` applies its augmentations
    there, such as the resize and the normalization. A
    `LuxonisLoaderTorch` without a height or a width returns the image
    unchanged. A loader that does not override the method raises
    ``NotImplementedError``. The function adds the batch dimension,
    moves the channels first, and casts to ``torch.float32``. It
    then runs `LuxonisLightningModule.full_forward` on this batch of
    one, under the ``image_source`` name of the module. The visualizers
    draw on the denormalized image. The function gives no labels, so no
    loss and no metric runs.

    Args:
        model (LuxonisModel): The model to run.
        images (``dict[str, Tensor]``): The raw image of shape
            ``[H, W, C]``, keyed by the input name of the loader.

    Returns:
        LuxonisOutput: ``outputs`` holds the packet of every output
        node. ``visualizations`` holds the image batch of every
        visualizer, with a batch size of ``1``. ``losses`` and
        ``metrics`` are empty.

    """
    npy_img = model.loaders["val"].augment_test_image(images)
    torch_img = torch.tensor(npy_img).unsqueeze(0).permute(0, 3, 1, 2).float()

    return model.lightning_module.full_forward(
        {model.lightning_module.image_source: torch_img},
        images=get_denormalized_images(model.cfg, torch_img),
        compute_visualizations=True,
    )


def window_closed() -> bool:  # pragma: no cover
    """Wait for a key press in the OpenCV windows and report a stop
    request.

    The function blocks in ``cv2.waitKey`` until the user presses a
    key.

    Returns:
        bool: ``True`` when the key is ``Esc`` or ``q``.

    """
    return cv2.waitKey(0) in {27, ord("q")}


def infer_from_video(
    model: "lxt.LuxonisModel", video_path: PathType, save_dir: Path | None
) -> None:
    """Run the model on every frame of a video, one frame at a time.

    The function reads the frames with ``cv2.VideoCapture``. When
    ``trainer.preprocessing.color_space`` of the config is ``"RGB"``,
    it converts each frame from BGR to RGB first. It then runs
    `prepare_and_infer_image` on the frame under the input name
    ``"image"``, and renders the visualizations with
    `process_visualizations`. When OpenCV cannot open ``video_path``,
    the function reads no frame and writes nothing.

    With ``save_dir``, the function writes the renders of each
    visualizer to ``<save_dir>/<node>_<visualizer>.mp4``. The video
    gets the ``mp4v`` codec, the frame rate of the source video, and
    the size of the first render. Without ``save_dir``, the function
    shows the
    render of each visualizer in an OpenCV window named
    ``<node>/<visualizer>``. It then waits for a key press after every
    frame. ``Esc`` or ``q`` stops the loop. At the end, the function
    releases the capture and the writers, and closes the windows.

    Args:
        model (LuxonisModel): The model to run.
        video_path (``PathType``): The video file.
        save_dir (``Path | None``): The directory of the output videos.
            ``None`` shows the renders on screen instead.

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
    """Run the prediction loop of the trainer over a loader and render
    the visualizations.

    The function runs ``model.pl_trainer.predict`` with
    ``model.lightning_module`` on ``loader``, so each batch goes
    through `LuxonisLightningModule.predict_step`.

    With ``save_dir``, a temporary ``BasePredictionWriter`` callback
    saves the renders of each batch as PNG files as they arrive:

    - ``<save_dir>/<image stem>_<node>_<visualizer>.png`` when
      ``img_paths`` is given. The callback takes the next path for
      each sample, in loader order.
    - ``<save_dir>/<node>_<visualizer>_<n>.png`` otherwise, where
      ``<n>`` counts the written files from ``0``.

    A ``/`` in a file name becomes ``-``.

    Without ``save_dir``, the function collects all predictions first.
    It then shows every render of every sample in an OpenCV window
    named ``<node>/<visualizer>``. It waits for a key press after
    every sample. ``Esc`` or ``q`` stops the loop. The function closes
    the windows at the end.

    Args:
        model (LuxonisModel): The model to run.
        loader (torch.utils.data.DataLoader): The batches to run on.
        save_dir (``PathType | None``): The directory of the PNG files.
            ``None`` shows the renders on screen instead.
        img_paths (``list[PathType] | None``): The source path of
            every sample, in loader order. It names the saved files.
            It is not read without ``save_dir``.

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
    """Callback that saves the renders of every prediction batch as PNG
    files, as `infer_from_loader` describes.
    """

    def __init__(
        self,
        save_dir: Path,
        img_paths: list[PathType] | None = None,
    ) -> None:
        super().__init__(write_interval="batch")
        self._save_dir = save_dir
        self._img_paths = img_paths
        self._counter = Counter()

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
            self._save_dir,
            self._counter,
            self._img_paths,
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
    """Build a ``DataLoader`` over image files.

    The function creates a local ``LuxonisDataset`` named
    ``infer_from_directory``, after it deletes a local dataset of that
    name. It adds every image with the sample metadata
    ``{"path": <image path>}``, and puts all images in the ``test``
    split. It wraps the dataset in a `LuxonisLoaderTorch` on the
    ``test`` view. The loader applies these settings of
    ``trainer.preprocessing`` of ``model``: the train image size, the
    active augmentations, the color space, and ``keep_aspect_ratio``.

    The loader does not keep the order of ``img_paths``. Use the
    ``"path"`` metadata to match a sample to its file. The function
    does not delete the dataset. `infer_from_directory` and
    `annotate_from_directory` delete it after use.

    Args:
        img_paths (``Iterable[PathType]``): The image files.
        model (LuxonisModel): The model whose preprocessing the loader
            applies.
        batch_size (int | None): The batch size. ``None`` selects
            ``trainer.batch_size`` of the config of ``model``.
        return_sample_metadata (bool): Also return the metadata of
            every sample. Each batch is then a tuple of the inputs,
            the labels, and a list with the metadata dictionary of
            every sample, each with the ``"path"`` key.

    Returns:
        torch.utils.data.DataLoader: The loader, with ``pin_memory``
        on and without shuffling. Without ``return_sample_metadata``,
        each batch is a list of the inputs and the labels. The inputs
        are one ``Tensor`` of shape ``[B, C, H, W]``. The labels are an
        empty dictionary, because the dataset has no annotations.

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
    """Run the model on image files.

    The function builds a loader with `create_loader_from_directory`
    and runs `infer_from_loader` on it. It passes ``img_paths`` in the
    given order to name the saved renders. The loader does not keep
    that order, so with two or more images a saved render can get the
    name of a different image. At the end, the function deletes the
    temporary local dataset ``infer_from_directory``.

    Args:
        model (LuxonisModel): The model to run.
        img_paths (``Iterable[PathType]``): The image files.
        save_dir (``Path | None``): The directory of the PNG files.
            ``None`` shows the renders on screen instead.

    """
    img_paths = list(img_paths)

    loader = create_loader_from_directory(img_paths, model)

    infer_from_loader(model, loader, save_dir, img_paths)
    inner_loader = cast(LuxonisLoaderTorch, loader.dataset)

    inner_loader.dataset.delete_dataset(delete_local=True)


class _LimitedLoader:
    """Wrapper of a ``DataLoader`` that yields at most ``n_batches``
    batches.
    """

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
    """Run the model on one view of its dataset.

    The function reads ``model.pytorch_loaders[view]``. When
    ``trainer.overfit_batches`` of the config is above ``0`` and
    ``view`` is ``"train"``, it reads only that many batches and logs
    a warning. It then runs `infer_from_loader` without image paths,
    so a saved render is named ``<node>_<visualizer>_<n>.png``.

    Args:
        model (LuxonisModel): The model to run.
        view (``Literal["train", "val", "test"]``): The dataset view to
            read.
        save_dir (``PathType | None``): The directory of the PNG files.
            ``None`` shows the renders on screen instead.

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
