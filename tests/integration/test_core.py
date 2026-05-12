import sqlite3
import sys
from pathlib import Path
from typing import Any, Literal

import pytest
import torch
from luxonis_ml.data import LuxonisDataset
from luxonis_ml.data.datasets.utils import shutil
from luxonis_ml.typing import Params, PathType

from luxonis_train.core import LuxonisModel


@pytest.mark.skipif(
    sys.platform == "win32", reason="Tuning not supported on Windows"
)
@pytest.mark.parametrize("monitor", ["loss", "metric"])
def test_tune(
    monitor: Literal["loss", "metric"],
    opts: Params,
    coco_dataset: LuxonisDataset,
    tmp_path: Path,
):
    study_path = tmp_path / "study.db"

    opts |= {
        "tuner.storage.database": str(study_path),
        "tuner.monitor": monitor,
        "tuner.n_trials": 4,
        "tuner.params": {
            "trainer.optimizer.name_categorical": ["Adam", "SGD"],
            "trainer.optimizer.params.lr_float": [0.0001, 0.001],
            "trainer.batch_size_int": [4, 16, 4],
            "trainer.preprocessing.augmentations_subset": [
                ["Defocus", "Sharpen", "Flip", "Normalize", "invalid"],
                2,
            ],
            "model.predefined_model.params.loss_params.weight_uniform": [
                0.1,
                0.9,
            ],
            "model.predefined_model.params.backbone_params.freezing.unfreeze_after_loguniform": [
                0.1,
                0.9,
            ],
        },
        "loader.params.dataset_name": coco_dataset.identifier,
    }
    model = LuxonisModel("configs/example_tuning.yaml", opts)
    model.tune()
    assert study_path.exists()
    con = sqlite3.connect(study_path)
    cur = con.cursor()
    cur.execute("SELECT * FROM trial_params")
    # Should be 4 * 6 = 24, but the augmentation
    # subset parameters are not stored in the database
    assert len(cur.fetchall()) == 20


def test_weights_loading(cifar10_dataset: LuxonisDataset, opts: Params):
    config_file = "configs/classification_light_model.yaml"
    opts |= {
        "loader.params.dataset_name": cifar10_dataset.dataset_name,
    }

    model = LuxonisModel(config_file, opts)
    test_results = model.test()
    assert test_results == model.test(
        weights={"state_dict": model.lightning_module.state_dict()}
    )


def test_checkpoint(
    tmp_path: Path, opts: Params, coco_dataset: LuxonisDataset
):
    def check_ckpt(path: Path | dict[str, Any]) -> None:
        if isinstance(path, dict):
            ckpt = path
        else:
            assert path.exists()
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
        assert "config" in ckpt
        assert "state_dict" in ckpt
        assert "version" in ckpt
        assert "dataset_metadata" in ckpt

    model = LuxonisModel(
        "configs/detection_light_model.yaml",
        opts
        | {
            "loader.params.dataset_name": coco_dataset.identifier,
            "trainer.callbacks": [
                {
                    "name": "UploadCheckpoint",
                    "active": True,
                },
            ],
        },  # type: ignore
    )
    model.test()

    ckpt = model.get_checkpoint()
    check_ckpt(ckpt)

    ckpt_path = model.save_checkpoint(tmp_path / "saved.ckpt")
    check_ckpt(ckpt_path)

    model.export(tmp_path / "exported.ckpt", ckpt_only=True)
    check_ckpt(tmp_path / "exported.ckpt")

    def upload_artifact(path: PathType, *args, **kwargs) -> None:
        path = Path(path)
        if path.suffix == ".ckpt":
            shutil.copy(path, tmp_path / "uploaded.ckpt")

    model.lightning_module.logger.upload_artifact = upload_artifact
    model.train()
    assert (tmp_path / "uploaded.ckpt").exists()
    check_ckpt(tmp_path / "uploaded.ckpt")


def test_precision_fallback_to_bf16_on_cpu(
    cifar10_dataset: LuxonisDataset, opts: Params
):
    opts |= {
        "loader.params.dataset_name": cifar10_dataset.dataset_name,
        "trainer.precision": "16-mixed",
        "trainer.accelerator": "cpu",
    }

    model = LuxonisModel("configs/classification_light_model.yaml", opts)
    model.test()
