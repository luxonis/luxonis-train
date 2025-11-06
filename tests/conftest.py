import random
import shutil
import zipfile
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import cv2
import gdown
import numpy as np
import numpy.typing as npt
import pytest
import torchvision
from _pytest.config import Config
from _pytest.python import Function
from luxonis_ml.data import Category, DatasetIterator, LuxonisDataset
from luxonis_ml.data.parsers import LuxonisParser
from luxonis_ml.typing import Params
from luxonis_ml.utils import LuxonisFileSystem, environ

from luxonis_train.config.config import OnnxExportConfig


@pytest.fixture(scope="session")
def work_dir() -> Generator[Path]:
    path = Path("tests", "work").absolute()
    path.mkdir(parents=True, exist_ok=True)

    yield path

    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture(scope="session")
def data_dir() -> Path:
    path = Path("tests", "data")
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture(scope="session")
def save_dir(work_dir: Path) -> Path:
    path = work_dir / "save-directory"
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture(scope="session")
def image_size() -> tuple[int, int]:
    return 32, 64


@pytest.fixture(scope="session", autouse=True)
def set_environment(work_dir: Path) -> None:
    environ.LUXONISML_BASE_PATH = work_dir / "luxonisml"


@pytest.fixture(scope="session")
def parking_lot_dataset(data_dir: Path) -> LuxonisDataset:
    url = "gs://luxonis-test-bucket/luxonis-train-test-data/datasets/ParkingLot3.zip"
    return LuxonisParser(
        url,
        dataset_name="ParkingLot3",
        delete_local=True,
        save_dir=data_dir,
    ).parse()


@pytest.fixture(scope="session")
def dinov3_weights() -> Path:
    checkpoint_name = "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
    dest_dir = Path("tests", "data", "checkpoints")
    remote_path = f"gs://luxonis-test-bucket/luxonis-train-test-data/checkpoints/{checkpoint_name}"
    return LuxonisFileSystem.download(remote_path, dest=dest_dir)


class LuxonisTestDataset(LuxonisDataset):
    def __init__(self, *args, source_path: Path, **kwargs):
        super().__init__(*args, **kwargs)
        self.source_path = source_path


@pytest.fixture(scope="session")
def embedding_dataset(data_dir: Path) -> LuxonisTestDataset:
    img_dir = data_dir / "embedding_images"
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

    dataset = LuxonisTestDataset(
        "embedding_test", delete_local=True, source_path=img_dir
    )
    dataset.add(generator())
    dataset.make_splits()
    return dataset


@pytest.fixture(scope="session")
def toy_ocr_dataset(data_dir: Path) -> LuxonisTestDataset:
    def generator() -> DatasetIterator:
        path = data_dir / "toy_ocr"
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

    dataset = LuxonisTestDataset(
        "toy_ocr", delete_local=True, source_path=data_dir / "toy_ocr"
    )
    dataset.add(generator())

    dataset.make_splits()
    return dataset


@pytest.fixture(scope="session")
def coco_dir(data_dir: Path) -> Path:
    url = "https://drive.google.com/uc?id=1XlvFK7aRmt8op6-hHkWVKIJQeDtOwoRT"
    coco_zip = data_dir / "COCO_people_subset.zip"

    if (
        not coco_zip.exists()
        and not (data_dir / "COCO_people_subset").exists()
    ):
        gdown.download(url, str(coco_zip), quiet=False)

    with zipfile.ZipFile(coco_zip, "r") as zip:
        unzip_dir = coco_zip.parent / coco_zip.stem
        zip.extractall(unzip_dir)
        return unzip_dir


@pytest.fixture(scope="session")
def coco_dataset(coco_dir: Path) -> LuxonisTestDataset:
    parser = LuxonisParser(
        str(coco_dir),
        dataset_name="coco_test",
        delete_local=True,
        dataset_plugin="LuxonisTestDataset",
        source_path=coco_dir / "person_val2017_subset",
    )
    return cast(LuxonisTestDataset, parser.parse(random_split=True))


@pytest.fixture(scope="session")
def cifar10_dataset(data_dir: Path) -> LuxonisTestDataset:
    output_folder = data_dir / "cifar10"
    if not output_folder.exists() or not list(output_folder.iterdir()):
        output_folder = LuxonisFileSystem.download(
            "gs://luxonis-test-bucket/luxonis-train-test-data/datasets/cifar10",
            data_dir,
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

    dataset = LuxonisTestDataset(
        "cifar10_test", delete_local=True, source_path=output_folder
    )
    dataset.add(CIFAR10_subset_generator())
    dataset.make_splits()
    return dataset


@pytest.fixture(scope="session")
def anomaly_detection_dataset(coco_dir: Path) -> LuxonisTestDataset:
    def random_square_mask(
        image_shape: tuple[int, int], n_squares: int = 1
    ) -> np.ndarray:
        mask = np.zeros(image_shape, dtype=np.uint8)
        h, w = image_shape
        rng = np.random.default_rng()
        for _ in range(n_squares):
            top_left = (
                rng.integers(0, w // 2),
                rng.integers(0, h // 2),
            )
            bottom_right = (
                rng.integers(w // 2, w),
                rng.integers(h // 2, h),
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

    paths_total = list((coco_dir).rglob("*.jpg"))
    train_paths = paths_total[:10]
    test_paths = paths_total[10:]

    dataset = LuxonisTestDataset(
        "dummy_mvtec",
        delete_local=True,
        source_path=coco_dir / "person_val2017_subset",
    )
    dataset.add(dummy_generator(train_paths, test_paths))
    definitions = {
        "train": train_paths,
        "val": test_paths[len(train_paths) // 2 :],
        "test": test_paths[: len(test_paths) // 2],
    }
    dataset.make_splits(definitions=definitions)
    return dataset


@pytest.fixture
def xor_dataset(data_dir: Path) -> LuxonisTestDataset:
    data_dir = data_dir / "xor_data"
    data_dir.mkdir(parents=True, exist_ok=True)

    def generator() -> DatasetIterator:
        inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]

        outputs = [0, 1, 1, 0]

        for i, (x_values, y_value) in enumerate(
            zip(inputs, outputs, strict=True)
        ):
            img_array = cast(
                npt.NDArray[np.uint8],
                np.array(x_values, dtype=np.uint8).reshape(1, 2) * 255,
            )
            img_path = data_dir / f"xor_{i}.png"
            cv2.imwrite(str(img_path), img_array)

            record = {
                "file": str(img_path),
                "annotation": {"class": f"xor_{y_value}"},
            }

            yield record

    dataset = LuxonisTestDataset(
        "xor_dataset", delete_local=True, source_path=data_dir
    )
    dataset.add(generator())
    dataset.make_splits((1, 0, 0))
    return dataset


@dataclass
class LuxonisTestDatasets:
    parking_lot_dataset: LuxonisDataset
    coco_dataset: LuxonisTestDataset
    cifar10_dataset: LuxonisTestDataset
    toy_ocr_dataset: LuxonisTestDataset
    embedding_dataset: LuxonisTestDataset
    anomaly_detection_dataset: LuxonisTestDataset


@pytest.fixture(scope="session")
def test_datasets(
    parking_lot_dataset: LuxonisDataset,
    coco_dataset: LuxonisTestDataset,
    cifar10_dataset: LuxonisTestDataset,
    toy_ocr_dataset: LuxonisTestDataset,
    embedding_dataset: LuxonisTestDataset,
    anomaly_detection_dataset: LuxonisTestDataset,
) -> LuxonisTestDatasets:
    return LuxonisTestDatasets(
        parking_lot_dataset,
        coco_dataset,
        cifar10_dataset,
        toy_ocr_dataset,
        embedding_dataset,
        anomaly_detection_dataset,
    )


@pytest.fixture
def opts(save_dir: Path, image_size: tuple[int, int]) -> Params:
    return {
        "trainer.epochs": 1,
        "trainer.batch_size": 2,
        "trainer.validation_interval": 1,
        "trainer.callbacks": [
            {"name": "TestOnTrainEnd", "active": False},
            {"name": "ExportOnTrainEnd", "active": False},
            {"name": "ArchiveOnTrainEnd", "active": False},
            {"name": "UploadCheckpoint", "active": False},
        ],
        "tracker.save_directory": str(save_dir),
        "trainer.preprocessing.train_image_size": image_size,
    }


@pytest.fixture
def current_opset() -> int:
    return OnnxExportConfig().opset_version


def pytest_collection_modifyitems(items: list[Function]):
    for item in items:
        path = str(item.fspath)
        if "/unittests/" in path:
            item.add_marker(pytest.mark.unit)
            item.add_marker(pytest.mark.order(0))
        elif "test_predefined_models.py" in path:
            item.add_marker(pytest.mark.predefined)
            item.add_marker(pytest.mark.order(2))
        elif "test_combinations.py" in path:
            item.add_marker(pytest.mark.combinations)
            item.add_marker(pytest.mark.order(3))
        else:
            item.add_marker(pytest.mark.misc)
            item.add_marker(pytest.mark.order(1))


def pytest_configure(config: Config):
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line(
        "markers", "predefined: mark test as a predefined model test"
    )
    config.addinivalue_line(
        "markers", "combinations: mark test as a combinations test"
    )
    config.addinivalue_line(
        "markers", "misc: mark test as a miscellaneous test"
    )
