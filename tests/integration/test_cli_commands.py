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
    tmp_path: Path,
    subtests: SubTests,
    opts: Params,
    save_dir: Path,
) -> None:
    flat_opts = []
    for key, value in opts.items():
        flat_opts.append(key)
        flat_opts.append(str(value))

    for command, kwargs in [
        (train, {}),
        (tune, {}),
        (_test, {"view": "val"}),
        (export, {"save_path": tmp_path}),
        (_yield_visualizations, {}),
        (archive, {"executable": tmp_path / "export.onnx"}),
    ]:
        with subtests.test(command.__name__):
            res = command(
                [
                    "loader.params.dataset_name",
                    coco_dataset.identifier,
                    "model.name",
                    command.__name__,
                    "tracker.run_name",
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
    with subtests.test("reload-from-checkpoint"):
        ckpt = next((save_dir / "train").rglob("*.ckpt"), None)
        assert ckpt is not None, "No checkpoint found after training."
        _test(
            [
                "loader.params.dataset_name",
                coco_dataset.identifier,
                "model.name",
                "test-reload-from-checkpoint",
                *flat_opts,
            ],
            weights=str(ckpt),
        )


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
    with pytest.raises(Exception):  # noqa: B017, PT011
        command(
            ["loader.params.dataset_name", coco_dataset.identifier],
            **kwargs,
        )


def test_source(tmp_path: Path, coco_dataset: LuxonisDataset):
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
    with open(tmp_path / "config.yaml", "w") as f:
        yaml.dump(cfg, f)

    with open(tmp_path / "source_1.py", "w") as f:
        f.write("print('sourcing 1')")

    with open(tmp_path / "source_2.py", "w") as f:
        f.write("print('sourcing 2')")

    with open(tmp_path / "callbacks.py", "w") as f:
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
    with open(tmp_path / "loss.py", "w") as f:
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
        [
            sys.executable,
            "-m",
            "luxonis_train",
            "--source",
            str(tmp_path / "source_1.py"),
            "test",
            "--source",
            str(tmp_path / "source_2.py"),
            "--config",
            str(tmp_path / "config.yaml"),
            "--source",
            str(tmp_path / "callbacks.py"),
            "--source",
            str(tmp_path / "loss.py"),
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
        check=False,
    )

    assert res.returncode == 0, res.stderr

    assert res.stdout is not None
    assert "sourcing 1" in res.stdout
    assert "sourcing 2" in res.stdout
    assert "custom callback message" in res.stdout
