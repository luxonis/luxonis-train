import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest
from luxonis_ml.data import LuxonisDataset
from luxonis_ml.typing import Kwargs
from luxonis_ml.utils import environ

from luxonis_train.__main__ import archive, export, inspect, train
from luxonis_train.__main__ import test as _test

ONNX_PATH = Path("tests/integration/client_commands_test_model.onnx")


@pytest.fixture(scope="session", autouse=True)
def prepare():
    os.environ["LUXONISML_BASE_PATH"] = str(environ.LUXONISML_BASE_PATH)
    yield
    ONNX_PATH.unlink(missing_ok=True)


@pytest.mark.parametrize(
    ("command", "kwargs"),
    [
        (train, {"config": "tests/configs/config_simple.yaml"}),
        (
            _test,
            {
                "config": "tests/configs/config_simple.yaml",
                "view": "val",
            },
        ),
        (
            export,
            {
                "config": "tests/configs/config_simple.yaml",
                "save_path": ONNX_PATH.parent,
            },
        ),
        (
            archive,
            {
                "config": "tests/configs/config_simple.yaml",
                "executable": ONNX_PATH,
            },
        ),
    ],
)
def test_cli_command_success(
    command: Callable, kwargs: Kwargs, coco_dataset: LuxonisDataset
) -> None:
    command(["loader.params.dataset_name", coco_dataset.identifier], **kwargs)


@pytest.mark.parametrize(
    ("command", "kwargs"),
    [
        (train, {"config": "nonexistent.yaml"}),
        (
            _test,
            {"config": "tests/configs/config_simple.yaml", "view": "invalid"},
        ),
        (export, {"config": "nonexistent.yaml"}),
        (
            inspect,
            {
                "config": "nonexistent.yaml",
                "view": "train",
                "size_multiplier": -1.0,
            },
        ),
        (archive, {"config": "nonexistent.yaml"}),
    ],
)
def test_cli_command_failure(
    command: Callable, kwargs: Kwargs, coco_dataset: LuxonisDataset
) -> None:
    with pytest.raises(Exception):  # noqa: PT011
        command(
            ["loader.params.dataset_name", coco_dataset.identifier],
            **kwargs,
        )


def test_source(work_dir: Path, coco_dataset: LuxonisDataset):
    with open(work_dir / "source_1.py", "w") as f:
        f.write("print('sourcing 1')")

    with open(work_dir / "source_2.py", "w") as f:
        f.write("print('sourcing 2')")

    with open(work_dir / "callbacks.py", "w") as f:
        f.write(
            """
import lightning.pytorch as pl

from luxonis_train import LuxonisLightningModule
from luxonis_train.registry import CALLBACKS


@CALLBACKS.register()
class CustomCallback(pl.Callback):
    def __init__(self, message: str, **kwargs):
        super().__init__(**kwargs)
        self.message = message

    def on_train_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: LuxonisLightningModule,
    ) -> None:
        print(self.message)
"""
        )
    with open(work_dir / "loss.py", "w") as f:
        f.write(
            """
from torch import Tensor

from luxonis_train import BaseLoss, Tasks

class CustomLoss(BaseLoss):
    supported_tasks = [Tasks.CLASSIFICATION, Tasks.SEGMENTATION]

    def __init__(self, smoothing: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.smoothing = smoothing

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        return predictions.sum() * self.smoothing
"""
        )

    result = subprocess.run(
        [  # noqa: S607
            sys.executable,
            "-m",
            "luxonis_train",
            "--source",
            str(work_dir / "source_1.py"),
            "test",
            "--source",
            str(work_dir / "source_2.py"),
            "--config",
            "tests/configs/config_simple.yaml",
            "--source",
            str(work_dir / "callbacks.py"),
            "--source",
            str(work_dir / "loss.py"),
            "loader.params.dataset_name",
            coco_dataset.identifier,
            "model.losses.0.name",
            "CustomLoss",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=os.environ
        | {
            "LUXONISML_BASE_PATH": str(environ.LUXONISML_BASE_PATH),
            "PYTHONIOENCODING": "utf-8",
        },
    )

    assert result.returncode == 0, (
        f"Command failed with return code {result.returncode}:\n{result.stderr}"
    )

    assert result.stdout is not None, "No stdout captured from subprocess"
    assert "sourcing 1" in result.stdout, "'sourcing 1' not found in output"
    assert "sourcing 2" in result.stdout, "'sourcing 2' not found in output"
