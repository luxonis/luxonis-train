import os
import subprocess
from pathlib import Path

from luxonis_ml.data import LuxonisDataset
from luxonis_ml.utils import environ


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
            "python",
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
        env=os.environ | {"LUXONISML_BASE_PATH": environ.LUXONISML_BASE_PATH},
    )

    assert result.returncode == 0, (
        f"Command failed with error: {result.stderr}"
    )

    assert "sourcing 1" in result.stdout
    assert "sourcing 2" in result.stdout
    assert "sourcing 3" in result.stdout
