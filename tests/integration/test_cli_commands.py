import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from types import GeneratorType

import pytest
import yaml
from luxonis_ml.data import LuxonisDataset
from luxonis_ml.typing import Kwargs, Params
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

    for command, kwargs in [
        (train, {}),
        (tune, {}),
        (_test, {"view": "val"}),
        (export, {"save_path": tempdir}),
        (_yield_visualizations, {}),
        (archive, {"executable": tempdir / "export.onnx"}),
    ]:
        with subtests.test(command.__name__):
            res = command(
                [
                    "loader.params.dataset_name",
                    coco_dataset.identifier,
                    "model.name",
                    command.__name__,
                    *flat_opts,
                ],
                config="configs/detection_light_model.yaml"
                if command is not tune
                else "configs/example_tuning.yaml",
                **kwargs,
            )
            if isinstance(res, GeneratorType):
                list(res)


@pytest.mark.parametrize(
    ("command", "kwargs"),
    [
        (train, {"config": "nonexistent.yaml"}),
        (
            _test,
            {
                "config": "configs/segmentation_light_model.yaml",
                "view": "invalid",
            },
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


def test_source(tempdir: Path, coco_dataset: LuxonisDataset):
    cfg = {
        "rich_logging": False,
        "model": {
            "name": "test_source",
            "nodes": [
                {
                    "name": "ResNet",
                },
                {
                    "name": "ClassificationHead",
                    "losses": [{"name": "CustomLoss"}],
                    "metrics": [{"name": "Accuracy"}],
                },
            ],
        },
        "trainer": {
            "callbacks": [
                {
                    "name": "CustomCallback",
                    "params": {"message": "custom callback message"},
                }
            ],
        },
    }
    with open(tempdir / "config.yaml", "w") as f:
        yaml.dump(cfg, f)

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
    def __init__(self, message: str, **kwargs):
        super().__init__(**kwargs)
        self.message = message

    def teardown(
        self,
        trainer: pl.Trainer,
        pl_module: LuxonisLightningModule,
        stage: str
    ) -> None:
        print(self.message)
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

    res = subprocess.run(
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
            str(tempdir / "config.yaml"),
            "--source",
            str(tempdir / "callbacks.py"),
            "--source",
            str(tempdir / "loss.py"),
            "loader.params.dataset_name",
            coco_dataset.identifier,
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

    assert res.returncode == 0, res.stderr

    assert res.stdout is not None
    assert "sourcing 1" in res.stdout
    assert "sourcing 2" in res.stdout
    assert "custom callback message" in res.stdout
