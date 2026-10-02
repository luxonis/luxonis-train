import signal
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import pytest

from luxonis_train.callbacks.graceful_interrupt import (
    GracefulInterruptCallback,
)
from luxonis_train.core import LuxonisModel


def test_graceful_interrupt_ignores_non_fit_stages(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    signal_calls: list[tuple[int, object]] = []
    monkeypatch.setattr(
        signal,
        "signal",
        lambda signum, handler: signal_calls.append((signum, handler)),
    )

    callback = GracefulInterruptCallback(tmp_path)
    callback.setup(Mock(), Mock(), stage="predict")

    assert signal_calls == []
    assert callback._signal_handlers == {}


def test_graceful_interrupt_restores_handlers_after_fit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    original_handlers = {
        signal.SIGINT: object(),
        signal.SIGTERM: object(),
    }
    signal_calls: list[tuple[int, object]] = []

    monkeypatch.setattr(signal, "getsignal", original_handlers.__getitem__)
    monkeypatch.setattr(
        signal,
        "signal",
        lambda signum, handler: signal_calls.append((signum, handler)),
    )

    callback = GracefulInterruptCallback(tmp_path)
    trainer = Mock()
    pl_module = Mock()

    callback.setup(trainer, pl_module, stage="fit")
    callback.teardown(trainer, pl_module, stage="fit")

    assert signal_calls == [
        (signal.SIGINT, callback._handle_signal),
        (signal.SIGTERM, callback._handle_signal),
        (signal.SIGINT, original_handlers[signal.SIGINT]),
        (signal.SIGTERM, original_handlers[signal.SIGTERM]),
    ]
    assert callback._signal_handlers == {}


def test_graceful_interrupt_uploads_the_checkpoint_and_keeps_the_run_open(
    tmp_path: Path,
) -> None:
    tracker = Mock()
    trainer = Mock()
    callback = GracefulInterruptCallback(tmp_path, tracker)
    callback.setup(trainer, Mock(), stage="validate")

    callback._handle_signal(signal.SIGINT, None)

    ckpt_path = tmp_path / "resume.ckpt"
    trainer.save_checkpoint.assert_called_once_with(ckpt_path)
    tracker.upload_artifact.assert_called_once_with(
        ckpt_path, typ="checkpoints", name="resume.ckpt"
    )
    # the training closes the run after it uploads the log and the config
    tracker.close.assert_not_called()
    assert trainer.should_stop is True


def test_an_interrupted_training_ends_as_failed() -> None:
    model = SimpleNamespace(pl_trainer=Mock(), _end_stage=Mock())
    model.pl_trainer.fit.side_effect = SystemExit(0)

    with pytest.raises(SystemExit):
        LuxonisModel._train(cast(LuxonisModel, model), None)

    model._end_stage.assert_called_once_with("failed")
