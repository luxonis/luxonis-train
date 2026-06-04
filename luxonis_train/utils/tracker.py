from lightning.pytorch.loggers.logger import Logger
from lightning.pytorch.utilities import rank_zero_only
from luxonis_ml.tracker import LuxonisTracker


class LuxonisTrackerPL(LuxonisTracker, Logger):
    """`LuxonisTracker` implementation compatible with PyTorch
    Lightning.
    """

    def __init__(self, *, _auto_finalize: bool = True, **kwargs):
        """Initialize the PyTorch Lightning tracker adapter.

        Args:
            _auto_finalize (bool): If ``True``, finalize the run automatically
                when training ends. If ``False``, the user must call
                `_finalize` manually.
            **kwargs (Any): Additional keyword arguments passed to
                `LuxonisTracker`.

        """
        LuxonisTracker.__init__(self, **kwargs)
        Logger.__init__(self)
        if _auto_finalize:
            self.finalize = self._finalize

    @rank_zero_only
    def _finalize(self, status: str = "success") -> None:  # pragma: no cover
        """Finalize current run.

        Args:
            status (str): Final run status. Defaults to ``"success"``.

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
