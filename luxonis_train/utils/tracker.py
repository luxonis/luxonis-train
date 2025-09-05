from lightning.pytorch.loggers.logger import Logger
from lightning.pytorch.utilities import rank_zero_only
from luxonis_ml.tracker import LuxonisTracker


class LuxonisTrackerPL(LuxonisTracker, Logger):
    """Implementation of LuxonisTracker that is compatible with
    PytorchLightning."""

    def __init__(self, *, _auto_finalize: bool = True, **kwargs):
        """
        @type _auto_finalize: bool
        @param _auto_finalize: If True, the run will be finalized automatically when the training ends.
            If set to C{False}, the user will have to call the L{_finalize} method manually.

        @type kwargs: dict
        @param kwargs: Additional keyword arguments to be passed to the L{LuxonisTracker}.
        """
        LuxonisTracker.__init__(self, **kwargs)
        Logger.__init__(self)
        if _auto_finalize:
            self.finalize = self._finalize

    @rank_zero_only
    def _finalize(self, status: str = "success") -> None:  # pragma: no cover
        """Finalizes current run."""
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
