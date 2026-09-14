"""Pre-annotation of a directory of images with a trained model, into a
new dataset.
"""

from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import torch
import torch.utils.data as torch_data
from loguru import logger
from luxonis_ml.data import DatasetIterator, LuxonisDataset
from luxonis_ml.data.datasets import DatasetRecord
from luxonis_ml.typing import PathType
from pydantic import ValidationError
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
    """Annotate image files with a model into a new dataset.

    The function runs these steps:

    - It builds a loader over ``img_paths`` with
      `create_loader_from_directory`, with a batch size of ``1``. That
      function replaces a local dataset named ``infer_from_directory``.
    - It creates the dataset ``dataset_name`` and adds the records of
      `annotated_dataset_generator` to it.
    - It splits a non-empty dataset into ``train``, ``val``, and
      ``test`` in the ratio 0.8, 0.1, and 0.1. For an empty dataset, it
      logs a warning.
    - It deletes the local copy of the ``infer_from_directory``
      dataset.

    The function does not load weights. `LuxonisModel.annotate` loads
    them before the call.

    Args:
        model (LuxonisModel): The model that predicts the annotations.
            The loader applies its ``trainer.preprocessing``.
        img_paths (``Iterable[PathType]``): The image files to annotate.
        dataset_name (str): The name of the new dataset.
        bucket_storage (``Literal["local", "gcs"]``): The storage
            backend of the new dataset.
        delete_local (bool): Delete the local files of an existing
            dataset named ``dataset_name`` before the function creates
            the new dataset. With ``False`` and ``"local"`` storage, the
            records go into the existing dataset.
        delete_remote (bool): Delete the remote files of an existing
            dataset named ``dataset_name``. The value has an effect only
            with ``"gcs"`` storage.
        team_id (str | None): The team that owns the dataset. ``None``
            reads ``LUXONISML_TEAM_ID`` from the environment.

    Returns:
        ``LuxonisDataset``: The new dataset with the annotations.

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
    """Yield the dataset records that the heads of a model predict.

    The generator puts the Lightning module of ``model`` in eval mode.
    For each batch of ``loader``, it runs
    `LuxonisLightningModule.full_forward` without gradients. For each
    output node that is a `BaseHead`, it calls `BaseHead.annotate`. The
    call gets the outputs of the head, the image paths from the
    ``"path"`` sample metadata, and the ``trainer.preprocessing`` of the
    config. The generator skips the other output nodes.

    A record from ``annotate`` that is a dictionary becomes a
    ``DatasetRecord``. When the only validation error of a record is a
    bounding box outside the clipping range, the generator skips the
    record and logs a debug message.

    Args:
        model (LuxonisModel): The model that predicts the annotations.
        loader (torch.utils.data.DataLoader): A loader whose batches hold
            the inputs, the labels, and a list with the metadata of each
            sample, as `create_loader_from_directory` builds with
            ``return_sample_metadata=True``.

    Yields:
        ``DatasetRecord``: The records that the heads give for the
        images.

    Raises:
        ValidationError: When a record fails the validation of
            ``DatasetRecord`` for another reason.

    """
    lt_module = model.lightning_module.eval()

    for imgs, _, sample_metadata in loader:
        with torch.no_grad():
            batch_out = lt_module.full_forward(imgs).outputs

        for head_name, head_output in batch_out.items():
            yield from _annotated_records(
                lt_module.nodes[head_name].module,
                head_output,
                [Path(meta["path"]) for meta in sample_metadata],
                model.cfg_preprocessing,
            )


def _annotated_records(
    head: object,
    head_output: Packet[Tensor],
    paths: list[Path],
    preprocessing: "PreprocessingConfig",
) -> DatasetIterator:
    if not isinstance(head, lxt.BaseHead):
        return
    for record in head.annotate(head_output, paths, preprocessing):
        if isinstance(record, DatasetRecord):  # pragma: no cover
            yield record
            continue
        try:
            yield DatasetRecord(**record)
        except ValidationError as error:
            errors = error.errors(include_url=False)
            if (
                len(errors) != 1
                or errors[0]["loc"] != ("annotation", "boundingbox")
                or "BBox annotation has value outside of automatic clipping range"
                not in errors[0]["msg"]
            ):
                raise
            logger.debug(
                "Skipping annotation with an out-of-range bounding box: {}",
                error,
            )
