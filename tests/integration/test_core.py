import sqlite3
import sys
from pathlib import Path
from typing import Literal

import pytest
from luxonis_ml.data import LuxonisDataset
from luxonis_ml.typing import Params

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
    weights = model.get_min_loss_checkpoint_path()
    assert test_results == model.test(weights=weights)


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
