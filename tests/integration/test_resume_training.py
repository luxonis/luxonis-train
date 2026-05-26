from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
from luxonis_ml.data import LuxonisDataset
from luxonis_ml.typing import Params

from luxonis_train.core import LuxonisModel


def _get_detection_light_train_opts(
    dataset_name: str,
    save_dir: Path,
    *,
    batch_size: int = 2,
    epochs: int = 1,
    overfit_batches: int = 1,
    resume_training: bool = False,
    strict_weights_loading: bool = False,
) -> Params:
    return {
        "loader.params.dataset_name": dataset_name,
        "loader.train_view": "train",
        "loader.val_view": "train",
        "loader.test_view": "train",
        "model.predefined_model.params.task_name": "vehicles",
        "trainer.overfit_batches": overfit_batches,
        "trainer.seed": 42,
        "trainer.deterministic": "warn",
        "trainer.epochs": epochs,
        "trainer.validation_interval": 1,
        "trainer.batch_size": batch_size,
        "trainer.resume_training": resume_training,
        "trainer.strict_weights_loading": strict_weights_loading,
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


def test_resume_training_with_ema_does_not_crash(
    parking_lot_dataset: LuxonisDataset,
    strict_loading_original_ckpt: Path,
    tmp_path: Path,
):
    config_file = "configs/detection_light_model.yaml"
    checkpoint = torch.load(
        strict_loading_original_ckpt, map_location="cpu", weights_only=False
    )
    resume_opts = _get_detection_light_train_opts(
        parking_lot_dataset.identifier,
        tmp_path / "save-directory",
        batch_size=4,
        epochs=int(checkpoint.get("epoch", 0)) + 2,
        overfit_batches=1,
        resume_training=True,
    )

    resumed_model = LuxonisModel(config_file, resume_opts)
    resumed_model.train(weights=strict_loading_original_ckpt)


def test_training_with_weights_without_resume_loads_checkpoint_only(
    parking_lot_dataset: LuxonisDataset,
    strict_loading_original_ckpt: Path,
    tmp_path: Path,
):
    config_file = "configs/detection_light_model.yaml"
    train_opts = _get_detection_light_train_opts(
        parking_lot_dataset.identifier,
        tmp_path / "save-directory",
    )

    fresh_model = LuxonisModel(config_file, train_opts)
    original_load_checkpoint = fresh_model.lightning_module.load_checkpoint
    fresh_model.lightning_module.load_checkpoint = Mock(
        wraps=original_load_checkpoint
    )
    fresh_model._train = Mock()

    fresh_model.train(weights=strict_loading_original_ckpt)

    fresh_model.lightning_module.load_checkpoint.assert_called_once()
    assert fresh_model.lightning_module.load_checkpoint.call_args is not None
    assert Path(
        fresh_model.lightning_module.load_checkpoint.call_args.args[0]
    ) == Path(strict_loading_original_ckpt)
    assert fresh_model._train.call_args is not None
    assert fresh_model._train.call_args.args[0] is None


def test_training_with_unexpected_checkpoint_key_fails_when_strict(
    parking_lot_dataset: LuxonisDataset,
    strict_loading_modified_model_ckpt: Path,
    tmp_path: Path,
):
    config_file = "configs/detection_light_model.yaml"
    train_opts = _get_detection_light_train_opts(
        parking_lot_dataset.identifier,
        tmp_path / "strict-save-directory",
        batch_size=4,
        epochs=1,
        overfit_batches=1,
        strict_weights_loading=True,
    )

    model = LuxonisModel(config_file, train_opts)

    with pytest.raises(RuntimeError, match="Unexpected key\\(s\\)"):
        model.train(weights=strict_loading_modified_model_ckpt)


def test_training_with_unexpected_checkpoint_key_succeeds_when_not_strict(
    parking_lot_dataset: LuxonisDataset,
    strict_loading_modified_model_ckpt: Path,
    tmp_path: Path,
):
    config_file = "configs/detection_light_model.yaml"
    train_opts = _get_detection_light_train_opts(
        parking_lot_dataset.identifier,
        tmp_path / "nonstrict-save-directory",
        batch_size=4,
        epochs=1,
        overfit_batches=1,
        strict_weights_loading=False,
    )

    model = LuxonisModel(config_file, train_opts)
    model.train(weights=strict_loading_modified_model_ckpt)


def test_strict_resume_loading_ignores_attached_module_checkpoint_keys(
    parking_lot_dataset: LuxonisDataset,
    strict_loading_modified_attached_modules_ckpt: Path,
    tmp_path: Path,
):
    config_file = "configs/detection_light_model.yaml"
    train_opts = _get_detection_light_train_opts(
        parking_lot_dataset.identifier,
        tmp_path / "attached-modules-save-directory",
        resume_training=True,
        strict_weights_loading=True,
    )
    model = LuxonisModel(config_file, train_opts)
    checkpoint = torch.load(
        strict_loading_modified_attached_modules_ckpt,
        map_location="cpu",
        weights_only=False,
    )

    incompatible = model.lightning_module.load_state_dict(
        checkpoint["state_dict"], strict=True
    )

    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
