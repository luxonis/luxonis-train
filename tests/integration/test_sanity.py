import os
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.fixture(scope="function", autouse=True)
def clear_output():
    shutil.rmtree("output", ignore_errors=True)


@pytest.mark.parametrize(
    "config_file", [path for path in os.listdir("configs") if "model" in path]
)
def test_sanity(config_file):
    opts = [
        "trainer.epochs",
        "1",
        "trainer.validation_interval",
        "1",
        "trainer.callbacks",
        "[]",
    ]
    result = subprocess.run(
        ["luxonis_train", "train", "--config", f"configs/{config_file}", *opts],
    )
    assert result.returncode == 0

    opts += ["model.weights", str(list(Path("output").rglob("*.ckpt"))[0])]
    opts += ["exporter.onnx.opset_version", "11"]

    result = subprocess.run(
        ["luxonis_train", "export", "--config", f"configs/{config_file}", *opts],
    )

    assert result.returncode == 0

    result = subprocess.run(
        ["luxonis_train", "eval", "--config", f"configs/{config_file}", *opts],
    )

    assert result.returncode == 0

    save_dir = Path("sanity_infer_save_dir")
    shutil.rmtree(save_dir, ignore_errors=True)

    result = subprocess.run(
        [
            "luxonis_train",
            "infer",
            "--save-dir",
            str(save_dir),
            "--config",
            f"configs/{config_file}",
            *opts,
        ],
    )

    assert result.returncode == 0
    assert save_dir.exists()
    assert len(list(save_dir.rglob("*.png"))) > 0
    shutil.rmtree(save_dir, ignore_errors=True)


def test_tuner():
    Path("study_local.db").unlink(missing_ok=True)
    result = subprocess.run(
        [
            "luxonis_train",
            "tune",
            "--config",
            "configs/example_tuning.yaml",
            "trainer.epochs",
            "1",
            "trainer.validation_interval",
            "1",
            "trainer.callbacks",
            "[]",
            "tuner.n_trials",
            "4",
        ],
    )
    assert result.returncode == 0
