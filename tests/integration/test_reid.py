import shutil
from pathlib import Path
from typing import Any

import pytest
import torch

from luxonis_train.attached_modules.losses.pml_loss import (
    ALL_EMBEDDING_LOSSES,
    CLASS_EMBEDDING_LOSSES,
)
from luxonis_train.core import LuxonisModel
from luxonis_train.enums import TaskType
from luxonis_train.loaders import BaseLoaderTorch

from .multi_input_modules import *

INFER_PATH = Path("tests/integration/infer-save-directory")
ONNX_PATH = Path("tests/integration/_model.onnx")
STUDY_PATH = Path("study_local.db")

NUM_INDIVIDUALS = 100


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
        image = torch.rand(self.input_shapes["image"], dtype=torch.float32)
        inputs = {
            "image": image,
        }

        # Fake labels
        id = torch.randint(0, NUM_INDIVIDUALS, (1,), dtype=torch.int64)
        labels = {
            "id": (id, TaskType.LABEL),
        }

        return inputs, labels

    def __len__(self):
        return 10

    def get_classes(self) -> dict[TaskType, list[str]]:
        return {TaskType.LABEL: ["id"]}


class CustomReIDLoaderNoID(CustomReIDLoader):
    def __getitem__(self, _):
        inputs, labels = super().__getitem__(_)
        labels["something_else"] = labels["id"]
        del labels["id"]

        return inputs, labels


class CustomReIDLoaderImageSize2(CustomReIDLoader):
    @property
    def input_shapes(self):
        return {
            "image": torch.Size([3, 200, 200]),
            "id": torch.Size([1]),
        }


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


not_class_based_losses = ALL_EMBEDDING_LOSSES.copy()
for loss in CLASS_EMBEDDING_LOSSES:
    not_class_based_losses.remove(loss)


@pytest.mark.parametrize("loss_name", not_class_based_losses)
def test_available_losses(
    opts: dict[str, Any], infer_path: Path, loss_name: str
):
    config_file = "tests/configs/reid.yaml"
    opts["model.losses.0.params.loss_name"] = loss_name

    # if loss_name in CLASS_EMBEDDING_LOSSES:
    #     opts["model.losses.0.params.num_classes"] = NUM_INDIVIDUALS
    #     opts["model.nodes.0.params.num_classes"] = NUM_INDIVIDUALS
    # else:
    #     opts["model.losses.0.params.num_classes"] = 0
    #     opts["model.nodes.0.params.num_classes"] = 0

    if loss_name == "RankedListLoss":
        opts["model.losses.0.params.loss_kwargs"] = {"margin": 1.0, "Tn": 0.5}

    model = LuxonisModel(config_file, opts)
    model.train()
    model.test(view="val")

    assert not ONNX_PATH.exists()
    model.export(str(ONNX_PATH))
    assert ONNX_PATH.exists()

    assert len(list(infer_path.iterdir())) == 0
    model.infer(view="val", save_dir=infer_path)
    assert infer_path.exists()


@pytest.mark.parametrize("loss_name", CLASS_EMBEDDING_LOSSES)
@pytest.mark.parametrize("num_classes", [-2, NUM_INDIVIDUALS])
def test_unsupported_class_based_losses(
    opts: dict[str, Any], loss_name: str, num_classes: int
):
    config_file = "tests/configs/reid.yaml"
    opts["model.losses.0.params.loss_name"] = loss_name
    opts["model.losses.0.params.num_classes"] = num_classes
    opts["model.nodes.0.params.num_classes"] = num_classes

    with pytest.raises(ValueError):
        LuxonisModel(config_file, opts)


@pytest.mark.parametrize("loss_name", ["NonExistentLoss"])
def test_nonexistent_losses(opts: dict[str, Any], loss_name: str):
    config_file = "tests/configs/reid.yaml"
    opts["model.losses.0.params.loss_name"] = loss_name

    with pytest.raises(ValueError):
        LuxonisModel(config_file, opts)


def test_bad_loader(opts: dict[str, Any]):
    config_file = "tests/configs/reid.yaml"
    opts["loader.name"] = "CustomReIDLoaderNoID"

    with pytest.raises(ValueError):
        model = LuxonisModel(config_file, opts)
        model.train()


def test_not_enough_samples_for_metrics(opts: dict[str, Any]):
    config_file = "tests/configs/reid.yaml"
    opts["model.metrics.1.params.cross_batch_memory_size"] = 100

    model = LuxonisModel(config_file, opts)
    model.train()


def test_image_size_not_divisible_by_32(opts: dict[str, Any]):
    config_file = "tests/configs/reid.yaml"
    opts["loader.name"] = "CustomReIDLoaderImageSize2"

    # with pytest.raises(ValueError):
    model = LuxonisModel(config_file, opts)
    model.train()
