import multiprocessing as mp
import os
import shutil
from pathlib import Path
from typing import Any

import cv2
import gdown
import numpy as np
import pytest
import torchvision
from luxonis_ml.data import Category, LuxonisDataset
from luxonis_ml.data.parsers import LuxonisParser
from luxonis_ml.utils import environ

WORK_DIR = Path("tests", "data").absolute()


@pytest.fixture(scope="session")
def output_dir() -> Path:
    return Path("tests/integration/save-directory")


@pytest.fixture(scope="session", autouse=True)
def setup(output_dir: Path):
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(WORK_DIR / "luxonisml", ignore_errors=True)
    shutil.rmtree(output_dir, ignore_errors=True)
    environ.LUXONISML_BASE_PATH = WORK_DIR / "luxonisml"
    output_dir.mkdir(exist_ok=True)


@pytest.fixture
def train_overfit() -> bool:
    return bool(os.getenv("LUXONIS_TRAIN_OVERFIT"))


@pytest.fixture(scope="session")
def parking_lot_dataset() -> LuxonisDataset:
    url = "gs://luxonis-test-bucket/luxonis-ml-test-data/D1_ParkingLot_Native.zip"
    parser = LuxonisParser(
        url,
        dataset_name="D1_ParkingLot",
        delete_existing=True,
        save_dir=WORK_DIR,
    )
    return parser.parse(random_split=True)


@pytest.fixture(scope="session")
def embedding_dataset() -> LuxonisDataset:
    img_dir = WORK_DIR / "embedding_images"
    img_dir.mkdir(exist_ok=True)

    def generator():
        for i in range(100):
            color = [(255, 0, 0), (0, 255, 0), (0, 0, 255)][i % 3]
            img = np.full((100, 100, 3), color, dtype=np.uint8)
            img[i, i] = 255
            cv2.imwrite(str(img_dir / f"image_{i}.png"), img)

            yield {
                "file": img_dir / f"image_{i}.png",
                "annotation": {
                    "metadata": {
                        "color": Category(["red", "green", "blue"][i % 3]),
                    },
                },
            }

    dataset = LuxonisDataset("embedding_test", delete_existing=True)
    dataset.add(generator())
    dataset.make_splits()
    return dataset


@pytest.fixture(scope="session")
def coco_dataset() -> LuxonisDataset:
    dataset_name = "coco_test"
    url = "https://drive.google.com/uc?id=1XlvFK7aRmt8op6-hHkWVKIJQeDtOwoRT"
    output_zip = WORK_DIR / "COCO_people_subset.zip"

    if (
        not output_zip.exists()
        and not (WORK_DIR / "COCO_people_subset").exists()
    ):
        gdown.download(url, str(output_zip), quiet=False)

    parser = LuxonisParser(
        str(output_zip), dataset_name=dataset_name, delete_existing=True
    )
    return parser.parse(random_split=True)


@pytest.fixture(scope="session")
def cifar10_dataset() -> LuxonisDataset:
    dataset = LuxonisDataset("cifar10_test", delete_existing=True)
    output_folder = WORK_DIR / "cifar10"
    output_folder.mkdir(parents=True, exist_ok=True)
    cifar10_torch = torchvision.datasets.CIFAR10(
        root=output_folder, train=False, download=True
    )
    classes = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]

    def CIFAR10_subset_generator():
        for i, (image, label) in enumerate(cifar10_torch):  # type: ignore
            if i == 1000:
                break
            path = output_folder / f"cifar_{i}.png"
            image.save(path)
            yield {
                "file": path,
                "annotation": {
                    "class": classes[label],
                },
            }

    dataset.add(CIFAR10_subset_generator())
    dataset.make_splits()
    return dataset


@pytest.fixture(scope="session")
def mnist_dataset() -> LuxonisDataset:
    dataset = LuxonisDataset("mnist_test", delete_existing=True)
    output_folder = WORK_DIR / "mnist"
    output_folder.mkdir(parents=True, exist_ok=True)
    mnist_torch = torchvision.datasets.MNIST(
        root=output_folder, train=False, download=True
    )

    def MNIST_subset_generator():
        for i, (image, label) in enumerate(mnist_torch):  # type: ignore
            if i == 1000:
                break
            path = output_folder / f"mnist_{i}.png"
            image.save(path)
            yield {
                "file": path,
                "annotation": {
                    "class": str(label),
                    "metadata": {"text": str(label)},
                },
            }

    dataset.add(MNIST_subset_generator())
    dataset.make_splits()
    return dataset


@pytest.fixture
def config(train_overfit: bool) -> dict[str, Any]:
    if train_overfit:  # pragma: no cover
        epochs = 100
    else:
        epochs = 1

    return {
        "tracker": {
            "save_directory": "tests/integration/save-directory",
        },
        "loader": {
            "train_view": "val",
            "params": {
                "dataset_name": "_ParkingLot",
            },
        },
        "trainer": {
            "batch_size": 4,
            "epochs": epochs,
            "n_workers": mp.cpu_count(),
            "validation_interval": epochs,
            "save_top_k": 0,
            "preprocessing": {
                "train_image_size": [256, 320],
                "keep_aspect_ratio": False,
                "normalize": {"active": True},
            },
            "callbacks": [
                {"name": "ExportOnTrainEnd"},
            ],
            "matmul_precision": "medium",
        },
    }
