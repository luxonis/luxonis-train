from pathlib import Path
from unittest.mock import Mock

import pytest
from lightning.pytorch.callbacks import ModelCheckpoint

from luxonis_train.callbacks.upload_checkpoint import UploadCheckpoint


def test_restored_upload_checkpoint_does_not_upload_known_path(tmp_path: Path):
    best_path = str(tmp_path / "min_val_loss" / "epoch.ckpt")
    callback = UploadCheckpoint()
    callback.load_state_dict({"last_best_checkpoints": {best_path}})

    model_checkpoint = Mock(spec=ModelCheckpoint)
    model_checkpoint.best_model_path = best_path
    trainer = Mock()
    trainer.checkpoint_callbacks = [model_checkpoint]
    module = Mock()
    checkpoint = {"callbacks": {callback.state_key: callback.state_dict()}}

    callback.on_save_checkpoint(trainer, module, checkpoint)

    module.logger.upload_artifact.assert_not_called()


def test_upload_checkpoint_persists_new_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    best_path = str(tmp_path / "best_val_metric" / "epoch.ckpt")
    callback = UploadCheckpoint()
    model_checkpoint = Mock(spec=ModelCheckpoint)
    model_checkpoint.best_model_path = best_path
    trainer = Mock()
    trainer.checkpoint_callbacks = [model_checkpoint]
    module = Mock()
    checkpoint = {"callbacks": {callback.state_key: callback.state_dict()}}

    callback.on_save_checkpoint(trainer, module, checkpoint)

    expected_state = {"last_best_checkpoints": {best_path}}
    assert checkpoint["callbacks"][callback.state_key] == expected_state
    module.logger.upload_artifact.assert_called_once_with(
        "best_val_metric.ckpt", typ="weights"
    )
