from pathlib import Path

import pytest
from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator,
)

from luxonis_train.utils import LuxonisTrackerPL


@pytest.mark.parametrize(
    ("auto_finalize", "steps"), [(True, [0]), (False, [0, 1])]
)
def test_lightning_finalize_closes_only_an_auto_finalized_run(
    tmp_path: Path, auto_finalize: bool, steps: list[int]
):
    tracker = LuxonisTrackerPL(
        run_name="run",
        save_directory=tmp_path,
        tensorboard=True,
        _auto_finalize=auto_finalize,
    )
    tracker.log_metrics({"loss": 1.0}, step=0)
    tracker.finalize("success")
    tracker.log_metrics({"loss": 0.5}, step=1)
    tracker.close()

    events = EventAccumulator(str(tmp_path / "tensorboard_logs" / "run"))
    events.Reload()
    assert [event.step for event in events.Scalars("loss")] == steps
