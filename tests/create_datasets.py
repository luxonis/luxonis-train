#!/usr/bin/env python
"""Recreate every dataset used by the test suite from scratch.

This mirrors the environment set up by the pytest fixtures in
``tests/conftest.py`` (same ``tests/work`` / ``tests/data`` directories and the
same ``LUXONISML_BASE_PATH``) and then builds all datasets by calling the
shared creation functions in ``tests/datasets.py``. Any existing dataset with
the same name is deleted before it is recreated. It is useful for warming the
local cache before running the tests offline, or for debugging dataset
generation in isolation.

Run it from the repository root:

    python -m tests.create_datasets

or, to (re)build only a subset:

    python -m tests.create_datasets coco cifar10
"""

from collections.abc import Callable
from pathlib import Path
from typing import Literal, get_args

from cyclopts import App
from luxonis_ml.data import LuxonisDataset
from luxonis_ml.utils import environ
from rich import print

from tests.datasets import (
    create_anomaly_detection_dataset,
    create_cifar10_dataset,
    create_coco_dataset,
    create_embedding_dataset,
    create_parking_lot_dataset,
    create_toy_ocr_dataset,
    create_xor_dataset,
    download_coco_dir,
)

DatasetName = Literal[
    "parking_lot",
    "embedding",
    "toy_ocr",
    "cifar10",
    "xor",
    "coco",
    "anomaly_detection",
]

DATASET_IDENTIFIERS: dict[str, str] = {
    "parking_lot": "ParkingLot3",
    "embedding": "embedding_test",
    "toy_ocr": "toy_ocr",
    "cifar10": "cifar10_test",
    "xor": "xor_dataset",
    "coco": "coco_test",
    "anomaly_detection": "dummy_mvtec",
}

DATASET_BUILDERS: dict[str, Callable[[Path], LuxonisDataset]] = {
    "parking_lot": create_parking_lot_dataset,
    "embedding": create_embedding_dataset,
    "toy_ocr": create_toy_ocr_dataset,
    "cifar10": create_cifar10_dataset,
    "xor": create_xor_dataset,
}

COCO_DATASETS = ("coco", "anomaly_detection")

ALL_DATASETS: tuple[DatasetName, ...] = get_args(DatasetName)

app = App(
    name="create-datasets",
    help="Recreate the datasets used by the test suite from scratch.",
)


def _setup_environment() -> Path:
    """Replicate the ``work_dir`` / ``data_dir`` / environment setup
    from the session-scoped fixtures in ``conftest.py`` and return the
    data directory.
    """
    work_dir = Path("tests", "work").absolute()
    work_dir.mkdir(parents=True, exist_ok=True)
    environ.LUXONISML_BASE_PATH = work_dir / "luxonisml"

    data_dir = Path("tests", "data")
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def _delete_existing(names: list[str]) -> None:
    for name in names:
        identifier = DATASET_IDENTIFIERS[name]
        if LuxonisDataset.exists(identifier):
            print(f"Deleting existing dataset '{identifier}' ...")
            LuxonisDataset(identifier).delete_dataset(delete_local=True)


def _build_coco_datasets(data_dir: Path) -> dict[str, LuxonisDataset]:
    """COCO and anomaly-detection share the same downloaded source
    directory.
    """
    coco_dir = download_coco_dir(data_dir)
    return {
        "coco": create_coco_dataset(coco_dir),
        "anomaly_detection": create_anomaly_detection_dataset(coco_dir),
    }


@app.default
def create_datasets(datasets: list[DatasetName] | None = None) -> None:
    """Delete any existing test datasets and recreate them from scratch.

    Args:
        datasets: Datasets to (re)create. Defaults to all of them.

    """
    names: list[str] = list(datasets) if datasets else list(ALL_DATASETS)

    data_dir = _setup_environment()
    _delete_existing(names)

    coco_needed = [name for name in names if name in COCO_DATASETS]
    coco_datasets: dict[str, LuxonisDataset] = {}
    if coco_needed:
        print(f"Building COCO-derived datasets: {', '.join(coco_needed)} ...")
        coco_datasets = _build_coco_datasets(data_dir)

    for name in names:
        if name in coco_datasets:
            dataset = coco_datasets[name]
        else:
            dataset = DATASET_BUILDERS[name](data_dir)
        print(f"  -> created '{dataset.identifier}'")

    print("Done.")


if __name__ == "__main__":
    app()
