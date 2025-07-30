import os
import subprocess
import sys
from pathlib import Path
from types import GeneratorType

import pytest
from luxonis_ml.data import LuxonisDataset
from luxonis_ml.typing import Params
from luxonis_ml.utils import environ
from pytest_subtests import SubTests

from luxonis_train.__main__ import (
    _yield_visualizations,
    archive,
    export,
    inspect,
    train,
    tune,
)
from luxonis_train.__main__ import test as _test


def test_cli_command_success(
    coco_dataset: LuxonisDataset,
    tempdir: Path,
    subtests: SubTests,
    opts: Params,
) -> None:
    flat_opts = []
    for key, value in opts.items():
        flat_opts.append(key)
        flat_opts.append(str(value))

    for cmd, kwargs in [
        (train, {}),
        (tune, {}),
        (_test, {"view": "val"}),
        (export, {"save_path": tempdir}),
        (_yield_visualizations, {}),
        (archive, {"executable": tempdir / "export.onnx"}),
    ]:
        with subtests.test(cmd.__name__):
            res = cmd(
                [
                    "loader.params.dataset_name",
                    coco_dataset.identifier,
                    "model.name",
                    cmd.__name__,
                    "tuner.n_trials",
                    "2",
                    "tuner.storage.database",
                    str(tempdir / "study_local.db"),
                    *flat_opts,
                ],
                config="configs/detection_light_model.yaml"
                if cmd is not tune
                else "configs/example_tuning.yaml",
                **kwargs,
            )
            if isinstance(res, GeneratorType):
                list(res)


def test_cli_command_failure(
    coco_dataset: LuxonisDataset,
    subtests: SubTests,
) -> None:
    for cmd, kwargs in [
        (train, {"config": "nonexistent.yaml"}),
        (train, {"weights": "nonexistent.ckpt"}),
        (_test, {"view": "invalid"}),
        (_test, {"weights": "nonexistent.ckpt"}),
        (export, {"weights": "nonexistent.ckpt"}),
        (
            inspect,
            {
                "view": "train",
                "size_multiplier": -1.0,
            },
        ),
        (archive, {"weights": "nonexistent.ckpt"}),
    ]:
        with subtests.test(cmd.__name__):
            with pytest.raises(Exception):  # noqa: PT011
                cmd(
                    ["loader.params.dataset_name", coco_dataset.identifier],
                    **kwargs,
                )


def test_source(tempdir: Path, cifar10_dataset: LuxonisDataset):
    with open(tempdir / "source_1.py", "w") as f:
        f.write("print('sourcing 1')")

    with open(tempdir / "source_2.py", "w") as f:
        f.write("print('sourcing 2')")

    with open(tempdir / "callbacks.py", "w") as f:
        f.write(
            """
import lightning.pytorch as pl

from luxonis_train import LuxonisLightningModule
from luxonis_train.registry import CALLBACKS


@CALLBACKS.register()
class CustomCallback(pl.Callback):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def teardown(
        self,
        trainer: pl.Trainer,
        pl_module: LuxonisLightningModule,
        stage: str,
    ) -> None:
        print("callback message")
"""
        )
    with open(tempdir / "loss.py", "w") as f:
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
            str(tempdir / "source_1.py"),
            "test",
            "--source",
            str(tempdir / "source_2.py"),
            "--config",
            "configs/classification_light_model.yaml",
            "--source",
            str(tempdir / "callbacks.py"),
            "--source",
            str(tempdir / "loss.py"),
            "loader.params.dataset_name",
            cifar10_dataset.identifier,
            "model.losses.0.name",
            "CustomLoss",
            "model.losses.0.attached_to",
            "ClassificationHead",
            "trainer.callbacks.3.name",
            "CustomCallback",
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
    assert "callback message" in result.stdout, (
        "'callback message' not found in output"
    )
