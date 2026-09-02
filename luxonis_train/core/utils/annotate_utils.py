from collections.abc import Iterable
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import torch
import torch.utils.data as torch_data
from loguru import logger
from luxonis_ml.data import DatasetIterator, LuxonisDataset
from luxonis_ml.data.datasets import DatasetRecord
from luxonis_ml.typing import PathType
from torch import Tensor

import luxonis_train as lxt
from luxonis_train.loaders.luxonis_loader_torch import LuxonisLoaderTorch
from luxonis_train.typing import Packet

from .infer_utils import create_loader_from_directory

if TYPE_CHECKING:
    from luxonis_train.config.config import PreprocessingConfig


def annotate_from_directory(
    model: "lxt.LuxonisModel",
    img_paths: Iterable[PathType],
    dataset_name: str,
    bucket_storage: Literal["local", "gcs"] = "local",
    delete_local: bool = True,
    delete_remote: bool = True,
    team_id: str | None = None,
) -> LuxonisDataset:
    """Annotate images from a directory using the specified model and
    create a LuxonisDataset.

    @param model: The LuxonisModel to use for annotation.
    @type model: lxt.LuxonisModel
    @param img_paths: Iterable of image paths to annotate.
    @type img_paths: Iterable[PathType]
    @param dataset_name: Name of the dataset to create.
    @type dataset_name: str
    @param bucket_storage: Storage type for the dataset, either 'local'
        or 'gcs'.
    @type bucket_storage: Literal['local', 'gcs']
    @param delete_local: Whether to delete local files after processing.
    @type delete_local: bool
    @param delete_remote: Whether to delete remote files after
        processing.
    @type delete_remote: bool
    @param team_id: Optional team ID for the dataset.
    @type team_id: str | None
    """
    img_paths = list(img_paths)

    loader = create_loader_from_directory(
        img_paths, model, return_sample_metadata=True, batch_size=1
    )

    annotated_dataset = LuxonisDataset(
        dataset_name=dataset_name,
        bucket_storage=bucket_storage,
        delete_local=delete_local,
        delete_remote=delete_remote,
        team_id=team_id,
    )

    generator = annotated_dataset_generator(model, loader)
    annotated_dataset.add(generator)
    if len(annotated_dataset) > 0:
        annotated_dataset.make_splits()
    else:
        logger.warning("No annotations were generated. The dataset is empty.")
    luxonis_loader = loader.dataset
    assert isinstance(luxonis_loader, LuxonisLoaderTorch)

    luxonis_loader.dataset.delete_dataset(delete_local=True)

    return annotated_dataset


def annotated_dataset_generator(
    model: "lxt.LuxonisModel", loader: torch_data.DataLoader
) -> DatasetIterator:
    """Create a generator that yields annotations for images processed
    by the model.
    """
    lt_module = model.lightning_module.eval()

    for imgs, _, sample_metadata in loader:
        with torch.no_grad():
            batch_out = lt_module.full_forward(imgs).outputs

        for head_name, head_output in batch_out.items():
            yield from _annotated_records(
                lt_module.nodes[head_name].module,
                head_output,
                [meta["path"] for meta in sample_metadata],
                model.cfg_preprocessing,
            )


def _annotated_records(
    head: object,
    head_output: Packet[Tensor],
    paths: Iterable[PathType],
    preprocessing: "PreprocessingConfig",
) -> DatasetIterator:
    if not isinstance(head, lxt.BaseHead):
        return
    for record in head.annotate(
        head_output, [Path(path) for path in paths], preprocessing
    ):
        if isinstance(record, DatasetRecord):  # pragma: no cover
            yield record
            continue
        # Skip predictions that are invalid, e.g. outside the clipping range.
        with suppress(Exception):
            yield DatasetRecord(**record)
