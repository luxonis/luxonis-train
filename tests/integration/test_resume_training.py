from pathlib import Path
from unittest.mock import Mock

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


def test_training_with_weights_without_resume_loads_checkpoint_only(
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

    fresh_model = LuxonisModel(config_file, train_opts)
    original_load_checkpoint = fresh_model.lightning_module.load_checkpoint
    fresh_model.lightning_module.load_checkpoint = Mock(
        wraps=original_load_checkpoint
    )
    fresh_model._train = Mock()

    fresh_model.train(weights=ckpt_path)

    fresh_model.lightning_module.load_checkpoint.assert_called_once_with(
        ckpt_path
    )
    assert fresh_model._train.call_args is not None
    assert fresh_model._train.call_args.args[0] is None
