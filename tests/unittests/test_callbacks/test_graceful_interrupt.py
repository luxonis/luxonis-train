import signal
from pathlib import Path
from unittest.mock import Mock, call

import pytest

from luxonis_train.callbacks.graceful_interrupt import (
    GracefulInterruptCallback,
)


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


def test_graceful_interrupt_uploads_the_checkpoint_and_closes_the_run(
    tmp_path: Path,
) -> None:
    tracker = Mock()
    callback = GracefulInterruptCallback(tmp_path, tracker)
    callback._trainer = Mock()

    callback._save_interrupt_checkpoint()

    checkpoint = tmp_path / "resume.ckpt"
    callback._trainer.save_checkpoint.assert_called_once_with(checkpoint)
    assert tracker.method_calls == [
        call.upload_artifact(
            checkpoint, typ="checkpoints", name="resume.ckpt"
        ),
        call.close(status="failed"),
    ]
