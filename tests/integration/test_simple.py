import json
import shutil
import sys
import tarfile
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from luxonis_ml.data import LuxonisDataset
from luxonis_ml.utils import environ

from luxonis_train.core import LuxonisModel

from .multi_input_modules import *

INFER_PATH = Path("tests/integration/infer-save-directory")
ONNX_PATH = Path("tests/integration/_model.onnx")
STUDY_PATH = Path("study_local.db")


@pytest.fixture
def opts(test_output_dir: Path) -> dict[str, Any]:
    return {
        "trainer.epochs": 1,
        "trainer.batch_size": 1,
        "trainer.validation_interval": 1,
        "trainer.callbacks": "[]",
        "tracker.save_directory": str(test_output_dir),
        "tuner.n_trials": 4,
    }


@pytest.fixture(scope="function", autouse=True)
def clear_files():
    # todo
    yield
    STUDY_PATH.unlink(missing_ok=True)
    ONNX_PATH.unlink(missing_ok=True)
    shutil.rmtree(INFER_PATH, ignore_errors=True)


@pytest.mark.parametrize(
    "config_file",
    [
        "classification_model",
        "segmentation_model",
        "detection_model",
        "keypoint_bbox_model",
    ],
)
def test_predefined_models(
    opts: dict[str, Any],
    config_file: str,
    coco_dataset: LuxonisDataset,
    cifar10_dataset: LuxonisDataset,
):
    config_file = f"configs/{config_file}.yaml"
    opts |= {
        "loader.params.dataset_name": cifar10_dataset.dataset_name
        if config_file == "classification_model"
        else coco_dataset.dataset_name,
    }
    model = LuxonisModel(config_file, opts)
    model.train()
    model.test()


def test_multi_input(opts: dict[str, Any]):
    config_file = "configs/example_multi_input.yaml"
    model = LuxonisModel(config_file, opts)
    model.train()
    model.test(view="val")

    assert not ONNX_PATH.exists()
    model.export(str(ONNX_PATH))
    assert ONNX_PATH.exists()

    assert not INFER_PATH.exists()
    model.infer(view="val", save_dir=INFER_PATH)
    assert INFER_PATH.exists()


def test_custom_tasks(
    opts: dict[str, Any], parking_lot_dataset: LuxonisDataset, subtests
):
    config_file = "tests/configs/parking_lot_config.yaml"
    opts |= {
        "loader.params.dataset_name": parking_lot_dataset.dataset_name,
        "trainer.batch_size": 2,
    }
    del opts["trainer.callbacks"]
    model = LuxonisModel(config_file, opts)
    model.train()
    archive_path = Path(
        model.run_save_dir, "archive", model.cfg.model.name
    ).with_suffix(".onnx.tar.xz")
    correct_archive_config = json.loads(
        Path("tests/integration/parking_lot.json").read_text()
    )

    with subtests.test("test_archive"):
        assert archive_path.exists()
        with tarfile.open(archive_path) as tar:
            extracted_cfg = tar.extractfile("config.json")

            assert (
                extracted_cfg is not None
            ), "Config JSON not found in the archive."
            generated_config = json.loads(extracted_cfg.read().decode())

        del generated_config["model"]["heads"][1]["metadata"]["anchors"]
        assert generated_config == correct_archive_config


@pytest.mark.skipif(
    environ.GOOGLE_APPLICATION_CREDENTIALS is None,
    reason="GCP credentials not set",
)
def test_parsing_loader():
    model = LuxonisModel("tests/configs/segmentation_parse_loader.yaml")
    model.train()


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Tuning not supported on Windows",
)
def test_tune(opts: dict[str, Any], coco_dataset: LuxonisDataset):
    opts["tuner.params"] = {
        "trainer.optimizer.name_categorical": ["Adam", "SGD"],
        "trainer.optimizer.params.lr_float": [0.0001, 0.001],
        "trainer.batch_size_int": [4, 16, 4],
        "trainer.preprocessing.augmentations_subset": [
            ["Defocus", "Sharpen", "Flip", "Normalize", "invalid"],
            2,
        ],
        "model.losses.0.weight_uniform": [0.1, 0.9],
        "model.nodes.0.freezing.unfreeze_after_loguniform": [0.1, 0.9],
    }
    opts["loader.params.dataset_name"] = coco_dataset.identifier
    model = LuxonisModel("configs/example_tuning.yaml", opts)
    model.tune()
    assert STUDY_PATH.exists()


def test_archive(test_output_dir: Path, coco_dataset: LuxonisDataset):
    opts = {
        "tracker.save_directory": str(test_output_dir),
        "loader.params.dataset_name": coco_dataset.identifier,
    }
    model = LuxonisModel("tests/configs/archive_config.yaml", opts)
    model.archive()
    assert (
        Path(
            model.run_save_dir,
            "archive",
            model.cfg.archiver.name or model.cfg.model.name,
        )
        .with_suffix(".onnx.tar.xz")
        .exists()
    )


def test_callbacks(opts: dict[str, Any], parking_lot_dataset: LuxonisDataset):
    config_file = "tests/configs/parking_lot_config.yaml"
    opts = deepcopy(opts)
    del opts["trainer.callbacks"]
    opts |= {
        "trainer.use_rich_progress_bar": False,
        "trainer.seed": 42,
        "trainer.deterministic": "warn",
        "trainer.callbacks": [
            {
                "name": "MetadataLogger",
                "params": {
                    "hyperparams": ["trainer.epochs", "trainer.batch_size"],
                },
            },
            {"name": "TestOnTrainEnd"},
            {"name": "UploadCheckpoint"},
            {
                "name": "ExportOnTrainEnd",
            },
            {
                "name": "ArchiveOnTrainEnd",
                "params": {"preferred_checkpoint": "loss"},
            },
        ],
        "exporter.scale_values": [0.5, 0.5, 0.5],
        "exporter.mean_values": [0.5, 0.5, 0.5],
        "exporter.blobconverter.active": True,
    }
    opts["loader.params.dataset_name"] = parking_lot_dataset.identifier
    model = LuxonisModel(config_file, opts)
    model.train()


def test_freezing(opts: dict[str, Any], coco_dataset: LuxonisDataset):
    config_file = "configs/segmentation_model.yaml"
    opts = deepcopy(opts)
    opts |= {
        "model.predefined_model.params": {
            "head_params": {
                "freezing": {
                    "active": True,
                    "unfreeze_after": 2,
                },
            }
        }
    }
    opts["trainer.epochs"] = 3
    opts["loader.params.dataset_name"] = coco_dataset.identifier
    model = LuxonisModel(config_file, opts)
    model.train()
