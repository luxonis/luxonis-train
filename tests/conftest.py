import shutil
from collections.abc import Generator
from pathlib import Path

import numpy as np
import pytest
from _pytest.config import Config
from _pytest.python import Function
from luxonis_ml.data import LuxonisDataset
from luxonis_ml.typing import Params
from luxonis_ml.utils import LuxonisFileSystem, environ
from PIL import Image

from luxonis_train.config.config import OnnxExportConfig

# Re-exported for backwards compatibility: several test modules import these
# names from ``tests.conftest``.
from tests.datasets import (
    LuxonisTestDataset,
    LuxonisTestDatasets,
    create_anomaly_detection_dataset,
    create_cifar10_dataset,
    create_coco_dataset,
    create_embedding_dataset,
    create_parking_lot_dataset,
    create_toy_ocr_dataset,
    create_xor_dataset,
    download_coco_dir,
)


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
    return create_parking_lot_dataset(data_dir)


@pytest.fixture(scope="session")
def dinov3_weights() -> Path:
    checkpoint_name = "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
    dest_dir = Path("tests", "data", "checkpoints")
    remote_path = f"gs://luxonis-test-bucket/luxonis-train-test-data/checkpoints/{checkpoint_name}"
    return LuxonisFileSystem.download(remote_path, dest=dest_dir)


@pytest.fixture(scope="session")
def strict_loading_original_ckpt() -> Path:
    checkpoint_name = "original_ckpt.ckpt"
    dest_dir = Path("tests", "data", "checkpoints")
    remote_path = f"gs://luxonis-test-bucket/luxonis-train-test-data/checkpoints/{checkpoint_name}"
    return LuxonisFileSystem.download(remote_path, dest=dest_dir)


@pytest.fixture(scope="session")
def strict_loading_modified_model_ckpt() -> Path:
    checkpoint_name = "modified_model_ckpt.ckpt"
    dest_dir = Path("tests", "data", "checkpoints")
    remote_path = f"gs://luxonis-test-bucket/luxonis-train-test-data/checkpoints/{checkpoint_name}"
    return LuxonisFileSystem.download(remote_path, dest=dest_dir)


@pytest.fixture(scope="session")
def strict_loading_modified_attached_modules_ckpt() -> Path:
    checkpoint_name = "modified_attached_modules_ckpt.ckpt"
    dest_dir = Path("tests", "data", "checkpoints")
    remote_path = f"gs://luxonis-test-bucket/luxonis-train-test-data/checkpoints/{checkpoint_name}"
    return LuxonisFileSystem.download(remote_path, dest=dest_dir)


@pytest.fixture(scope="session")
def embedding_dataset(data_dir: Path) -> LuxonisTestDataset:
    return create_embedding_dataset(data_dir)


@pytest.fixture(scope="session")
def toy_ocr_dataset(data_dir: Path) -> LuxonisTestDataset:
    return create_toy_ocr_dataset(data_dir)


@pytest.fixture(scope="session")
def coco_dir(data_dir: Path) -> Path:
    return download_coco_dir(data_dir)


@pytest.fixture(scope="session")
def coco_dataset(coco_dir: Path) -> LuxonisTestDataset:
    return create_coco_dataset(coco_dir)


@pytest.fixture(scope="session")
def cifar10_dataset(data_dir: Path) -> LuxonisTestDataset:
    return create_cifar10_dataset(data_dir)


@pytest.fixture(scope="session")
def anomaly_detection_dataset(coco_dir: Path) -> LuxonisTestDataset:
    return create_anomaly_detection_dataset(coco_dir)


@pytest.fixture
def xor_dataset(data_dir: Path) -> LuxonisTestDataset:
    return create_xor_dataset(data_dir)


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


@pytest.fixture(scope="session")
def embeddings_visualizer_references(
    data_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    remote_dir = "gs://luxonis-test-bucket/luxonis-train-test-data/reference_images/embeddings_visualizer"
    ref_dir = data_dir / "reference_images" / "embeddings_visualizer"
    ref_dir.mkdir(parents=True, exist_ok=True)

    kde_ref_path = ref_dir / "kdeplot.png"
    scatter_ref_path = ref_dir / "scatterplot.png"

    if not kde_ref_path.exists():
        LuxonisFileSystem.download(f"{remote_dir}/kdeplot.png", dest=ref_dir)
    if not scatter_ref_path.exists():
        LuxonisFileSystem.download(
            f"{remote_dir}/scatterplot.png", dest=ref_dir
        )

    kde_ref = np.array(Image.open(kde_ref_path).convert("RGB"))
    scatter_ref = np.array(Image.open(scatter_ref_path).convert("RGB"))

    return kde_ref, scatter_ref


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
            {"name": "ConvertOnTrainEnd", "active": False},
            {"name": "UploadCheckpoint", "active": False},
        ],
        "tracker.save_directory": str(save_dir),
        "trainer.preprocessing.train_image_size": image_size,
        "exporter.aimet": {
            "active": False,
            "epochs": 1,
            "fold_batch_norms": True,
            "batch_norm_reestimation": True,
            "cross_layer_equalization": True,
            "sequential_mse": True,
        },
        "exporter.aimet.adaround": {
            "active": True,
            "default_num_iterations": 1,
        },
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
        elif "test_overfit_convergence.py" in path:
            item.add_marker(pytest.mark.overfit_convergence)
            item.add_marker(pytest.mark.order(4))
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
    config.addinivalue_line(
        "markers",
        "overfit_convergence: mark test as an overfit convergence test",
    )
    config.addinivalue_line(
        "markers",
        "flaky(reruns, reruns_delay): rerun intermittently failing tests",
    )
