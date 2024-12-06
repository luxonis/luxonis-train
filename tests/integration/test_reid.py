import shutil
from pathlib import Path
from typing import Any

import pytest
import torch

from luxonis_train.core import LuxonisModel
from luxonis_train.enums import TaskType
from luxonis_train.loaders import BaseLoaderTorch

from .multi_input_modules import *

INFER_PATH = Path("tests/integration/infer-save-directory")
ONNX_PATH = Path("tests/integration/_model.onnx")
STUDY_PATH = Path("study_local.db")


class CustomReIDLoader(BaseLoaderTorch):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def input_shapes(self):
        return {
            "image": torch.Size([3, 256, 256]),
            "id": torch.Size([1]),
        }

    def __getitem__(self, _):  # pragma: no cover
        # Fake data
        image = torch.rand(3, 256, 256, dtype=torch.float32)
        inputs = {
            "image": image,
        }

        # Fake labels
        id = torch.randint(0, 1000, (1,), dtype=torch.int64)
        labels = {
            "id": (id, TaskType.LABEL),
        }

        return inputs, labels

    def __len__(self):
        return 10

    def get_classes(self) -> dict[TaskType, list[str]]:
        return {TaskType.LABEL: ["id"]}


@pytest.fixture
def infer_path() -> Path:
    if INFER_PATH.exists():
        shutil.rmtree(INFER_PATH)
    INFER_PATH.mkdir()
    return INFER_PATH


@pytest.fixture
def opts(test_output_dir: Path) -> dict[str, Any]:
    return {
        "trainer.epochs": 1,
        "trainer.batch_size": 2,
        "trainer.validation_interval": 1,
        "trainer.callbacks": "[]",
        "tracker.save_directory": str(test_output_dir),
        "tuner.n_trials": 4,
    }


@pytest.fixture(scope="function", autouse=True)
def clear_files():
    yield
    STUDY_PATH.unlink(missing_ok=True)
    ONNX_PATH.unlink(missing_ok=True)


def test_reid(opts: dict[str, Any], infer_path: Path):
    config_file = "tests/configs/reid.yaml"
    model = LuxonisModel(config_file, opts)
    model.train()
    model.test(view="val")

    assert not ONNX_PATH.exists()
    model.export(str(ONNX_PATH))
    assert ONNX_PATH.exists()

    assert len(list(infer_path.iterdir())) == 0
    model.infer(view="val", save_dir=infer_path)
    assert infer_path.exists()
