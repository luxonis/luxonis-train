from typing import Any

import pytest

from luxonis_train.utils import tracker as tracker_module


def test_luxonis_tracker_pl_initializes_bases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    def luxonis_tracker_init(self: Any, **kwargs: Any) -> None:
        calls.append(("tracker", kwargs))

    def logger_init(self: Any) -> None:
        calls.append(("logger", {}))

    monkeypatch.setattr(
        tracker_module.LuxonisTracker, "__init__", luxonis_tracker_init
    )
    monkeypatch.setattr(tracker_module.Logger, "__init__", logger_init)

    tracker = tracker_module.LuxonisTrackerPL(project_name="demo")

    assert calls == [("tracker", {"project_name": "demo"}), ("logger", {})]
    assert tracker.finalize == tracker._finalize


def test_luxonis_tracker_pl_can_skip_auto_finalize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def noop_init(self: Any) -> None:
        pass

    monkeypatch.setattr(tracker_module.LuxonisTracker, "__init__", noop_init)
    monkeypatch.setattr(tracker_module.Logger, "__init__", noop_init)

    tracker = tracker_module.LuxonisTrackerPL(_auto_finalize=False)

    assert "finalize" not in tracker.__dict__
