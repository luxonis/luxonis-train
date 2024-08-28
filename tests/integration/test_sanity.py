import json
import shutil
import sys
import tarfile
from copy import deepcopy
from pathlib import Path

import pytest
from luxonis_ml.data import LuxonisDataset
from multi_input_modules import *

from luxonis_train.core import LuxonisModel

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


@pytest.fixture(scope="session", autouse=True)
def manage_out_dir():
    shutil.rmtree(TEST_OUTPUT, ignore_errors=True)
    TEST_OUTPUT.mkdir(exist_ok=True)


@pytest.fixture(scope="function", autouse=True)
def clear_files():
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


def test_custom_tasks(parking_lot_dataset: LuxonisDataset, subtests):
    config_file = "tests/configs/parking_lot_config.yaml"
    opts = deepcopy(OPTS) | {
        "loader.params.dataset_name": parking_lot_dataset.dataset_name,
        "trainer.batch_size": 2,
    }
    del opts["trainer.callbacks"]
    model = LuxonisModel(config_file, opts=opts)
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

            assert extracted_cfg is not None, "Config JSON not found in the archive."
            generated_config = json.loads(extracted_cfg.read().decode())

        del generated_config["model"]["heads"][1]["metadata"]["anchors"]
        assert generated_config == correct_archive_config

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
