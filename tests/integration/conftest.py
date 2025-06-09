import multiprocessing as mp
import os
import random
import shutil
from copy import deepcopy
from pathlib import Path

import cv2
import gdown
import numpy as np
import pytest
import torchvision
from luxonis_ml.data import Category, DatasetIterator, LuxonisDataset
from luxonis_ml.data.parsers import LuxonisParser
from luxonis_ml.typing import Kwargs
from luxonis_ml.utils import LuxonisFileSystem, environ


@pytest.fixture(scope="session")
def work_dir() -> Path:
    return Path("tests", "data").absolute()


@pytest.fixture(scope="session")
def image_size() -> tuple[int, int]:
    return 64, 128


@pytest.fixture(scope="session")
def batch_size() -> int:
    return 2


@pytest.fixture(scope="session")
def output_dir() -> Path:
    return Path("tests/integration/save-directory")


@pytest.fixture(scope="session", autouse=True)
def setup(output_dir: Path, work_dir: Path):
    work_dir.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(work_dir / "luxonisml", ignore_errors=True)
    shutil.rmtree(output_dir, ignore_errors=True)
    environ.LUXONISML_BASE_PATH = work_dir / "luxonisml"
    output_dir.mkdir(exist_ok=True)


@pytest.fixture
def train_overfit() -> bool:
    return bool(os.getenv("LUXONIS_TRAIN_OVERFIT"))


@pytest.fixture(scope="session")
def parking_lot_dataset(work_dir: Path) -> LuxonisDataset:
    url = "gs://luxonis-test-bucket/luxonis-ml-test-data/D1_ParkingLot_Native.zip"
    parser = LuxonisParser(
        url,
        dataset_name="D1_ParkingLot",
        delete_local=True,
        save_dir=work_dir,
    )
    return parser.parse(random_split=True)


@pytest.fixture(scope="session")
def embedding_dataset(work_dir: Path) -> LuxonisDataset:
    img_dir = work_dir / "embedding_images"
    img_dir.mkdir(exist_ok=True)

    def generator() -> DatasetIterator:
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

    dataset = LuxonisDataset("embedding_test", delete_local=True)
    dataset.add(generator())
    dataset.make_splits()
    return dataset


@pytest.fixture(scope="session")
def toy_ocr_dataset(work_dir: Path) -> LuxonisDataset:
    def generator():
        path = work_dir / "toy_ocr"
        path.mkdir(parents=True, exist_ok=True)
        alphabet = "abcdefghijklmnopqrstuvwxyz"
        for _ in range(50):
            random_word = "".join(
                random.choices(alphabet, k=random.randint(3, 10))
            )
            img = np.full((64, 320, 3), 255, dtype=np.uint8)
            cv2.putText(
                img,
                random_word,
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imwrite(str(path / f"{random_word}.png"), img)
            yield {
                "file": path / f"{random_word}.png",
                "annotation": {
                    "metadata": {"text": random_word},
                },
            }

    dataset = LuxonisDataset("toy_ocr", delete_local=True)
    dataset.add(generator())

    dataset.make_splits()
    return dataset


@pytest.fixture(scope="session")
def coco_dataset(work_dir: Path) -> LuxonisDataset:
    dataset_name = "coco_test"
    url = "https://drive.google.com/uc?id=1XlvFK7aRmt8op6-hHkWVKIJQeDtOwoRT"
    output_zip = work_dir / "COCO_people_subset.zip"

    if (
        not output_zip.exists()
        and not (work_dir / "COCO_people_subset").exists()
    ):
        gdown.download(url, str(output_zip), quiet=False)

    parser = LuxonisParser(
        str(output_zip), dataset_name=dataset_name, delete_local=True
    )
    return parser.parse(random_split=True)


@pytest.fixture(scope="session")
def cifar10_dataset(work_dir: Path) -> LuxonisDataset:
    dataset = LuxonisDataset("cifar10_test", delete_local=True)
    output_folder = work_dir / "cifar10"
    if not os.path.exists(output_folder) or not os.listdir(output_folder):
        output_folder = LuxonisFileSystem.download(
            "gs://luxonis-test-bucket/luxonis-train-test-data/datasets/cifar10",
            work_dir,
        )
    cifar10_torch = torchvision.datasets.CIFAR10(
        root=output_folder, train=False, download=False
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

    def CIFAR10_subset_generator() -> DatasetIterator:
        for i, (image, label) in enumerate(cifar10_torch):  # type: ignore
            if i == 20:
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
def mnist_dataset(work_dir: Path) -> LuxonisDataset:
    dataset = LuxonisDataset("mnist_test", delete_local=True)
    output_folder = work_dir / "mnist"
    output_folder.mkdir(parents=True, exist_ok=True)
    mnist_torch = torchvision.datasets.MNIST(
        root=output_folder, train=False, download=True
    )

    def MNIST_subset_generator() -> DatasetIterator:
        for i, (image, label) in enumerate(mnist_torch):  # type: ignore
            if i == 20:
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


@pytest.fixture(scope="session")
def anomaly_detection_dataset(work_dir: Path) -> LuxonisDataset:
    url = "https://drive.google.com/uc?id=1XlvFK7aRmt8op6-hHkWVKIJQeDtOwoRT"
    output_zip = work_dir / "COCO_people_subset.zip"

    if (
        not output_zip.exists()
        and not (work_dir / "COCO_people_subset").exists()
    ):
        gdown.download(url, str(output_zip), quiet=False)

    def random_square_mask(
        image_shape: tuple[int, int], n_squares: int = 1
    ) -> np.ndarray:
        mask = np.zeros(image_shape, dtype=np.uint8)
        h, w = image_shape
        for _ in range(n_squares):
            top_left = (
                np.random.randint(0, w // 2),
                np.random.randint(0, h // 2),
            )
            bottom_right = (
                np.random.randint(w // 2, w),
                np.random.randint(h // 2, h),
            )
            cv2.rectangle(mask, top_left, bottom_right, 255, -1)
        return mask

    def dummy_generator(
        train_paths: list[Path], test_paths: list[Path]
    ) -> DatasetIterator:
        for path in train_paths:
            img = cv2.imread(str(path))
            img_h, img_w, _ = img.shape
            mask = np.zeros((img_h, img_w), dtype=np.uint8)
            yield {
                "file": path,
                "annotation": {
                    "class": "object",
                    "segmentation": {"mask": mask},
                },
            }

        for path in test_paths:
            img = cv2.imread(str(path))
            img_h, img_w, _ = img.shape
            mask = random_square_mask((img_h, img_w))
            poly = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )[0]
            poly_normalized = [
                [(x / img_w, y / img_h) for x, y in contour.reshape(-1, 2)]
                for contour in poly
            ]
            yield {
                "file": path,
                "annotation": {
                    "class": "object",
                    "segmentation": {
                        "height": img_h,
                        "width": img_w,
                        "points": [
                            pt for segment in poly_normalized for pt in segment
                        ],
                    },
                },
            }

    paths_total = list((work_dir / "COCO_people_subset/").rglob("*.jpg"))
    train_paths = paths_total[:5]
    test_paths = paths_total[5:]

    dataset = LuxonisDataset("dummy_mvtec", delete_local=True)
    dataset.add(dummy_generator(train_paths, test_paths))
    definitions = {
        "train": train_paths,
        "val": test_paths,
    }
    dataset.make_splits(definitions=definitions)
    return dataset


@pytest.fixture
def config(
    train_overfit: bool, image_size: tuple[int, int], batch_size: int
) -> Kwargs:
    if train_overfit:  # pragma: no cover
        epochs = 100
    else:
        epochs = 1

    return deepcopy(
        {
            "tracker": {
                "save_directory": "tests/integration/save-directory",
            },
            "loader": {
                "train_view": "val",
            },
            "trainer": {
                "batch_size": batch_size,
                "epochs": epochs,
                "n_workers": mp.cpu_count(),
                "validation_interval": epochs,
                "save_top_k": 0,
                "preprocessing": {
                    "train_image_size": image_size,
                    "keep_aspect_ratio": False,
                    "normalize": {"active": True},
                },
                "callbacks": [
                    {"name": "ExportOnTrainEnd"},
                ],
                "matmul_precision": "medium",
            },
        }
    )
