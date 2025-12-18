import time
from typing import Any

import lightning.pytorch as pl
from lightning.pytorch.utilities import rank_zero_only
from lightning.pytorch.utilities.types import STEP_OUTPUT
from loguru import logger
from typing_extensions import override

import luxonis_train as lxt


class TrainingProgressCallback(pl.Callback):
    """Publish training progress metrics to MLflow for HubAI
    integration.

    This callback only logs when MLflow is enabled. Metrics published:
        - C{train/epoch_progress_percent}: Percentage of current epoch completed
        - C{train/epoch_duration_sec}: Duration of completed epoch in seconds
    """

    def __init__(self, log_every_n_batches: int = 1):
        """
        @type log_every_n_batches: int
        @param log_every_n_batches: How often to log progress metrics
        (every N batches). Can be set to higher to prevent logging
        in real-time if there is too much logging overhead.
        By default 1 means real-time.
        """
        super().__init__()
        self.log_every_n_batches = max(1, log_every_n_batches)
        self._epoch_start_time: float | None = None
        self._is_mlflow: bool = False

    @override
    def setup(
        self,
        trainer: pl.Trainer,
        pl_module: "lxt.LuxonisLightningModule",
        stage: str | None = None,
    ) -> None:
        self._is_mlflow = pl_module.cfg.tracker.is_mlflow

    @override
    def on_train_epoch_start(
        self,
        trainer: pl.Trainer,
        pl_module: "lxt.LuxonisLightningModule",
    ) -> None:
        self._epoch_start_time = time.time()

        if not self._is_mlflow:
            logger.warning(
                "TrainingProgressCallback logs epoch-specific progress as MLFlow keys; please set is_mlflow to True in the tracker config to enable this."
            )
            return
        if trainer.logger is None:
            logger.warning(
                "TrainingProgressCallback requires a logger to be configured."
            )
            return

        trainer.logger.log_metrics(
            {
                "train/epoch_progress_percent": 0.0,
            },
            step=trainer.global_step,
        )

    @rank_zero_only
    @override
    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: "lxt.LuxonisLightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if not self._is_mlflow or trainer.logger is None:
            return

        # Log every N batches to reduce overhead
        if (batch_idx + 1) % self.log_every_n_batches != 0:
            return

        total_batches = trainer.num_training_batches
        current_batch = batch_idx + 1  # 1-indexed

        progress_percent = (
            (current_batch / total_batches) * 100 if total_batches > 0 else 0.0
        )

        trainer.logger.log_metrics(
            {
                "train/epoch_progress_percent": progress_percent,
            },
            step=trainer.global_step,
        )

    @rank_zero_only
    @override
    def on_train_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: "lxt.LuxonisLightningModule",
    ) -> None:
        if not self._is_mlflow or trainer.logger is None:
            return

        epoch_duration = (
            time.time() - self._epoch_start_time
            if self._epoch_start_time is not None
            else 0.0
        )

        trainer.logger.log_metrics(
            {
                "train/epoch_duration_sec": epoch_duration,
                "train/epoch_progress_percent": 100.0,
            },
            step=trainer.global_step,
        )
