import shutil
import sys
from pathlib import Path

import pytest
from luxonis_ml.data import LuxonisDataset
from multi_input_modules import *

from luxonis_train.core import LuxonisModel
from luxonis_train.utils.config import Config

TEST_OUTPUT = Path("tests/integration/_test-output")
INFER_PATH = Path("tests/integration/_infer_save_dir")
ONNX_PATH = Path("tests/integration/_model.onnx")
STUDY_PATH = Path("study_local.db")

OPTS = {
    "trainer.epochs": 1,
    "trainer.batch_size": 1,
    "trainer.validation_interval": 1,
    "trainer.callbacks": "[]",
    "tracker.save_directory": str(TEST_OUTPUT),
    "tuner.n_trials": 4,
}


@pytest.fixture(scope="function", autouse=True)
def clear_output():
    Config.clear_instance()
    shutil.rmtree(TEST_OUTPUT, ignore_errors=True)
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
        "resnet_model",
        "coco_model",
        "efficient_coco_model",
    ],
)
def test_simple_models(config_file: str):
    config_file = f"configs/{config_file}.yaml"
    model = LuxonisModel(config_file, opts=OPTS)
    model.train()
    model.test()
    model.export()
    assert (
        Path(model.run_save_dir, "export", model.cfg.model.name)
        .with_suffix(".onnx")
        .exists()
    )
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
    del model


def test_multi_input():
    config_file = "configs/example_multi_input.yaml"
    model = LuxonisModel(config_file, opts=OPTS)
    model.train()
    model.test(view="val")

    assert not ONNX_PATH.exists()
    model.export(str(ONNX_PATH))
    assert ONNX_PATH.exists()

    assert not INFER_PATH.exists()
    model.infer(view="val", save_dir=INFER_PATH)
    assert INFER_PATH.exists()
    del model


def test_custom_tasks(parking_lot_dataset: LuxonisDataset):
    config_file = "tests/configs/parking_lot_config.yaml"
    model = LuxonisModel(
        config_file,
        opts=OPTS
        | {
            "loader.params.dataset_name": parking_lot_dataset.dataset_name,
            "trainer.batch_size": 2,
        },
    )
    model.train()
    assert model.archive().exists()
    del model


def test_parsing_loader():
    model = LuxonisModel("tests/configs/segmentation_parse_loader.yaml")
    model.train()
    del model


@pytest.mark.skipif(sys.platform == "win32", reason="Tuning not supported on Windows")
def test_tuner():
    model = LuxonisModel("configs/example_tuning.yaml", opts=OPTS)
    model.tune()
    assert STUDY_PATH.exists()
    del model
