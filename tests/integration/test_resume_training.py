from pathlib import Path

from luxonis_ml.data import LuxonisDataset
from luxonis_ml.typing import Params

from luxonis_train.core import LuxonisModel


def test_resume_training_with_ema_does_not_crash(
    parking_lot_dataset: LuxonisDataset, opts: Params, tmp_path: Path
):
    config_file = "configs/detection_light_model.yaml"
    save_dir = tmp_path / "save-directory"

    train_opts = opts | {
        "loader.params.dataset_name": parking_lot_dataset.identifier,
        "loader.train_view": "train",
        "loader.val_view": "train",
        "loader.test_view": "train",
        "model.predefined_model.params.task_name": "vehicles",
        "trainer.overfit_batches": 1,
        "trainer.seed": 42,
        "trainer.deterministic": "warn",
        "trainer.epochs": 1,
        "trainer.validation_interval": 1,
        "tracker.save_directory": str(save_dir),
        "trainer.callbacks": [
            {
                "name": "EMACallback",
                "active": True,
                "params": {"decay": 0.9999},
            },
            {"name": "TestOnTrainEnd", "active": False},
            {"name": "ExportOnTrainEnd", "active": False},
            {"name": "ArchiveOnTrainEnd", "active": False},
            {"name": "ConvertOnTrainEnd", "active": False},
            {"name": "UploadCheckpoint", "active": False},
        ],
    }

    model = LuxonisModel(config_file, train_opts)
    model.train()

    ckpt_path = model.get_best_metric_checkpoint_path()
    assert ckpt_path, "No checkpoint found after initial training"

    resume_opts = train_opts | {
        "trainer.resume_training": True,
        "trainer.epochs": 2,
    }
    resumed_model = LuxonisModel(config_file, resume_opts)
    resumed_model.train(weights=ckpt_path)
