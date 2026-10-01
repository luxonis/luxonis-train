"""The experiment tracker for PyTorch Lightning, over the backends of
``luxonis_ml.tracker``, such as TensorBoard, Weights and Biases, and
MLFlow.
"""

from typing import Any

from lightning.pytorch.loggers.logger import Logger
from luxonis_ml.tracker import LuxonisTracker


class LuxonisTrackerPL(LuxonisTracker, Logger):
    """Lightning logger built on ``luxonis_ml.tracker.LuxonisTracker``.

    The class adds the ``Logger`` interface of Lightning to the tracker
    of ``luxonis_ml``. A ``Trainer`` can then log to each backend of the
    tracker through it. `LuxonisModel` creates one tracker for each run.

    """

    def __init__(self, *, _auto_finalize: bool = True, **kwargs):
        """Initialize the tracker and the Lightning logger.

        Args:
            _auto_finalize: Whether the ``Trainer`` closes the run. With
                ``True``, the instance replaces ``finalize`` with
                ``close``. The ``Trainer`` calls ``finalize("success")``
                at the end of each ``fit``, ``validate``, ``test``, or
                ``predict`` call, and ``finalize("failed")`` on an
                exception. With ``False``, ``finalize`` of Lightning
                stays, and the run closes at ``close`` or when the
                process exits.
            **kwargs: Keyword arguments for
                ``luxonis_ml.tracker.LuxonisTracker``, such as
                ``project_name``, ``run_name``, ``save_directory``, and
                a keyword for each backend, such as ``mlflow``.

        """
        LuxonisTracker.__init__(self, **kwargs)
        Logger.__init__(self)
        if _auto_finalize:
            self.finalize = self.close


def get_tracker_init_params(cfg_tracker: Any) -> dict[str, Any]:
    """Build the keyword arguments of the tracker from its config.

    ``model_dump`` of `TrackerConfig` leaves out ``save_directory``, so
    the function adds it back. A plugin backend in the config becomes a
    keyword of its name.

    Args:
        cfg_tracker: The tracker config, a `TrackerConfig`. The function
            calls its ``model_dump`` and reads its ``save_directory``.

    Returns:
        The fields of ``cfg_tracker``, with ``save_directory``.
        `LuxonisModel` passes them to `LuxonisTrackerPL`.

    Example:
        >>> from luxonis_train.config.config import TrackerConfig
        >>> from luxonis_train.utils import get_tracker_init_params
        >>> config = TrackerConfig(run_name="baseline", my_service=True)
        >>> "save_directory" in config.model_dump()
        False
        >>> params = get_tracker_init_params(config)
        >>> params["run_name"], params["my_service"]
        ('baseline', True)
        >>> str(params["save_directory"])
        'output'

    """
    tracker_params = cfg_tracker.model_dump()
    tracker_params["save_directory"] = cfg_tracker.save_directory
    return tracker_params
