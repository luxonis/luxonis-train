from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy.typing as npt
from luxonis_ml.tracker import RunStatus, TrackerBackend
from luxonis_ml.typing import ParamValue

from luxonis_train.config.config import TrackerConfig, WandbTrackerConfig
from luxonis_train.utils.tracker import (
    LuxonisTrackerPL,
    get_tracker_init_params,
)


class StatusBackend(TrackerBackend, register_name="status_backend"):
    """Record the status that the run closes with."""

    status: RunStatus | None = None

    def start(self) -> None:
        pass

    def log_hyperparams(self, params: Mapping[str, ParamValue]) -> None:
        pass

    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None:
        pass

    def log_image(self, name: str, image: npt.NDArray[Any], step: int) -> None:
        pass

    def log_matrix(
        self,
        matrix: npt.NDArray[Any],
        name: str,
        step: int,
        extra_data: Mapping[str, ParamValue],
    ) -> None:
        pass

    def close(self, status: RunStatus) -> None:
        self.status = status


def make_tracker(tmp_path: Path, *, auto_finalize: bool) -> LuxonisTrackerPL:
    tracker = LuxonisTrackerPL(
        _auto_finalize=auto_finalize,
        run_name="0-test",
        save_directory=tmp_path,
        tensorboard=False,
        status_backend=True,
    )
    tracker.log_metrics({"loss": 0.5}, 1)
    return tracker


def closed_with(tracker: LuxonisTrackerPL) -> RunStatus | None:
    return tracker.get_backend(StatusBackend).status


def test_a_run_marked_as_failed_closes_as_failed(tmp_path: Path):
    tracker = make_tracker(tmp_path, auto_finalize=False)

    tracker.mark_failed()
    tracker.close("success")

    assert closed_with(tracker) == "failed"


def test_a_run_closes_with_its_status_without_a_failure(tmp_path: Path):
    tracker = make_tracker(tmp_path, auto_finalize=False)

    tracker.close("success")

    assert closed_with(tracker) == "success"


def test_lightning_closes_the_run_of_a_trial(tmp_path: Path):
    tracker = make_tracker(tmp_path, auto_finalize=True)

    tracker.finalize("success")

    assert closed_with(tracker) == "success"


def test_lightning_leaves_the_main_run_open(tmp_path: Path):
    """A later export or archive still uploads to the run."""
    tracker = make_tracker(tmp_path, auto_finalize=False)

    tracker.finalize("success")

    assert closed_with(tracker) is None
    tracker.close()


def test_the_config_becomes_the_arguments_of_the_tracker(tmp_path: Path):
    cfg = TrackerConfig(
        project_name="project",
        save_directory=tmp_path,
        tensorboard=False,
        wandb=WandbTrackerConfig(entity="team"),
        mlflow=True,
        plugins={"status_backend": True},
    )

    assert get_tracker_init_params(cfg) == {
        "project_name": "project",
        "project_id": None,
        "run_name": None,
        "run_id": None,
        "save_directory": tmp_path,
        "tensorboard": False,
        "wandb": {"entity": "team"},
        "mlflow": True,
        "status_backend": True,
    }
