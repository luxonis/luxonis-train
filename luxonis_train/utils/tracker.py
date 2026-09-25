"""The experiment tracker for PyTorch Lightning, over TensorBoard,
Weights and Biases, and MLFlow.
"""

from typing import Any

from lightning.pytorch.loggers.logger import Logger
from lightning.pytorch.utilities import rank_zero_only
from luxonis_ml.tracker import LuxonisTracker


class LuxonisTrackerPL(LuxonisTracker, Logger):
    """Lightning logger built on ``luxonis_ml.tracker.LuxonisTracker``.

    The class adds the ``Logger`` interface of Lightning to the tracker
    of ``luxonis_ml``. A ``Trainer`` can then log to TensorBoard,
    Weights and Biases, and MLFlow through it. `LuxonisModel` creates
    one tracker for each run.

    """

    def __init__(self, *, _auto_finalize: bool = True, **kwargs):
        """Initialize the tracker and the Lightning logger.

        Args:
            _auto_finalize (bool): Whether the ``Trainer`` closes the run.
                With ``True``, the instance replaces ``finalize`` with
                ``_finalize``. The ``Trainer`` calls
                ``finalize("success")`` at the end of each ``fit``,
                ``validate``, ``test``, or ``predict`` call. It calls
                ``finalize("failed")`` on an exception. With ``False``,
                ``finalize`` of Lightning stays, and the caller must call
                ``_finalize``. `LuxonisModel.finalize_run` does this.
            **kwargs (``Any``): Keyword arguments for
                ``luxonis_ml.tracker.LuxonisTracker``, such as
                ``project_name``, ``run_name``, ``save_directory``, and
                ``is_mlflow``.

        """
        LuxonisTracker.__init__(self, **kwargs)
        Logger.__init__(self)
        if _auto_finalize:
            self.finalize = self._finalize

    @rank_zero_only
    def _finalize(self, status: str = "success") -> None:  # pragma: no cover
        """Close the run on every active backend.

        The method runs on rank zero only. It flushes and closes
        TensorBoard. It ends the MLFlow run as ``FINISHED`` for
        ``"success"`` or ``"finished"`` and as ``FAILED`` otherwise, and
        then calls ``close``. It finishes Weights and Biases with the
        exit code ``0`` for ``"success"`` and ``1`` otherwise.

        Args:
            status (str): The final status of the run.

        """
        if self.is_tensorboard:
            self.experiment["tensorboard"].flush()
            self.experiment["tensorboard"].close()
        if self.is_mlflow:
            if status in ["success", "finished"]:
                mlflow_status = "FINISHED"
            else:
                mlflow_status = "FAILED"
            self.experiment["mlflow"].end_run(mlflow_status)
            self.close()
        if self.is_wandb:
            wandb_status = 0 if status == "success" else 1
            self.experiment["wandb"].finish(wandb_status)


def get_tracker_init_params(cfg_tracker: Any) -> dict[str, Any]:
    """Build the keyword arguments of the tracker from its config.

    ``model_dump`` of `TrackerConfig` leaves out ``save_directory``, so
    the function adds it back.

    Args:
        cfg_tracker (``Any``): The tracker config, a `TrackerConfig`.
            The function calls its ``model_dump`` and reads its
            ``save_directory``.

    Returns:
        ``dict[str, Any]``: The fields of ``cfg_tracker``, with
        ``save_directory``. `LuxonisModel` passes them to
        `LuxonisTrackerPL`.

    Example:
        >>> from luxonis_train.config.config import TrackerConfig
        >>> from luxonis_train.utils import get_tracker_init_params
        >>> config = TrackerConfig(run_name="baseline")
        >>> "save_directory" in config.model_dump()
        False
        >>> params = get_tracker_init_params(config)
        >>> params["run_name"], str(params["save_directory"])
        ('baseline', 'output')

    """
    tracker_params = cfg_tracker.model_dump()
    tracker_params["save_directory"] = cfg_tracker.save_directory
    return tracker_params
