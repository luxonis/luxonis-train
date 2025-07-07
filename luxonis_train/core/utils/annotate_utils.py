from collections.abc import Iterable
from pathlib import Path
from typing import Literal

import torch
import torch.utils.data as torch_data
from luxonis_ml.data import DatasetIterator, LuxonisDataset, LuxonisLoader
from luxonis_ml.typing import PathType

import luxonis_train as lxt
from luxonis_train.loaders.luxonis_loader_torch import LuxonisLoaderTorch

from .infer_utils import create_loader_from_directory


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
        img_paths, model, add_path_annotation=True
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
    annotated_dataset.make_splits()
    luxonis_loader = loader.dataset
    assert isinstance(luxonis_loader, LuxonisLoaderTorch)

    luxonis_loader.dataset.delete_dataset()

    return annotated_dataset


def annotated_dataset_generator(
    model: "lxt.LuxonisModel", loader: torch_data.DataLoader
) -> DatasetIterator:
    """Generator that yields annotations for images processed by the
    model."""

    lt_module = model.lightning_module.eval()

    for batch in loader:
        imgs, metas = batch
        with torch.no_grad():
            batch_out = lt_module(imgs).outputs

        for head_name, head_output in batch_out.items():
            img_paths = [
                Path("".join(chr(int(c.item())) for c in raw_meta))
                for raw_meta in metas["/metadata/path"]
            ]
            head = lt_module.nodes[head_name]
            if isinstance(head, lxt.BaseHead):
                yield from head.annotate(
                    head_output, img_paths, model.cfg_preprocessing
                )
