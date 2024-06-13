from lightning.pytorch.loggers.logger import Logger
from lightning.pytorch.utilities import rank_zero_only  # type: ignore
from luxonis_ml.tracker import LuxonisTracker


class LuxonisTrackerPL(LuxonisTracker, Logger):
    """Implementation of LuxonisTracker that is compatible with PytorchLightning."""

    @rank_zero_only
    def finalize(self, status: str = "success") -> None:
        """Finalizes current run."""
        if self.is_tensorboard:
            self.experiment["tensorboard"].flush()
            self.experiment["tensorboard"].close()
        if self.is_mlflow:
            if status == "success":
                mlflow_status = "FINISHED"
            elif status == "failed":
                mlflow_status = "FAILED"
            elif status == "finished":
                mlflow_status = "FINISHED"
            self.experiment["mlflow"].end_run(mlflow_status)
        if self.is_wandb:
            if status == "success":
                wandb_status = 0
            else:
                wandb_status = 1
            self.experiment["wandb"].finish(wandb_status)
