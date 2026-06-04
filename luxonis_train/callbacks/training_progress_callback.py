import time
from math import isfinite
from typing import Any

import lightning.pytorch as pl
from lightning.pytorch.utilities import rank_zero_only
from lightning.pytorch.utilities.types import STEP_OUTPUT
from loguru import logger
from typing_extensions import override

import luxonis_train as lxt


class TrainingProgressCallback(pl.Callback):
    """Publish training progress metrics.

    Metrics published:
        - ``train/epoch_progress_percent``: Percentage of current epoch completed
        - ``train/epoch_duration_sec``: Time elapsed so far in current epoch (updated per batch)
        - ``train/epoch_completion_sec``: Total duration of completed training epoch in seconds
        - ``train/batch_total_sec``: Time spent processing one training batch
        - ``val/epoch_progress_percent``: Percentage of current validation epoch completed
        - ``val/epoch_duration_sec``: Time elapsed so far in current validation epoch
        - ``val/epoch_completion_sec``: Total duration of completed validation epoch in seconds
        - ``val/batch_total_sec``: Time spent processing one validation batch
        - ``test/epoch_progress_percent``: Percentage of current test epoch completed
        - ``test/epoch_duration_sec``: Time elapsed so far in current test epoch
        - ``test/epoch_completion_sec``: Total duration of completed test epoch in seconds
        - ``test/batch_total_sec``: Time spent processing one test batch
    """

    def __init__(self, log_every_n_batches: int = 1):
        """Initialize training progress logging frequency.

        Args:
            log_every_n_batches (int): How often to log progress metrics (every N batches). Can be set to higher to prevent logging in real-time if there is too much logging overhead. By default 1 means real-time.

        """
        super().__init__()
        self.log_every_n_batches = max(1, log_every_n_batches)
        self._train_epoch_start_time: float | None = None
        self._val_epoch_start_time: float | None = None
        self._test_epoch_start_time: float | None = None
        self._train_batch_start_time: float | None = None
        self._val_batch_start_time: float | None = None
        self._test_batch_start_time: float | None = None
        self._train_batch_step = 0
        self._val_batch_step = 0
        self._test_batch_step = 0
        self._val_epoch_batch_count = 0
        self._test_epoch_batch_count = 0

    @staticmethod
    def _now() -> float:
        return time.perf_counter()

    @staticmethod
    def _elapsed(start_time: float | None) -> float:
        if start_time is None:
            return 0.0
        return time.perf_counter() - start_time

    @staticmethod
    def _total_batches(
        total_batches: float | list[int | float],
    ) -> int:
        """Return the total number of batches across eval
        dataloaders.
        """
        if isinstance(total_batches, list):
            return sum(
                int(batch_count)
                for batch_count in total_batches
                if isfinite(batch_count)
            )
        if not isfinite(total_batches):
            return 0
        return int(total_batches)

    @override
    def on_train_epoch_start(
        self,
        trainer: pl.Trainer,
        pl_module: "lxt.LuxonisLightningModule",
    ) -> None:
        self._train_epoch_start_time = self._now()

        if trainer.logger is None:
            logger.warning(
                "TrainingProgressCallback requires a logger to be configured."
            )
            return

        # Keep train progress/timing metrics on a cumulative batch axis.
        # `global_step` tracks optimizer steps, so with gradient
        # accumulation multiple train batches can collapse onto the same
        # step and stop being truly per-batch aligned.
        trainer.logger.log_metrics(
            {
                "train/epoch_progress_percent": 0.0,
            },
            step=self._train_batch_step,
        )

    @override
    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: "lxt.LuxonisLightningModule",
        batch: Any,
        batch_idx: int,
    ) -> None:
        self._train_batch_start_time = self._now()

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
        self._train_batch_step += 1

        if trainer.logger is None:
            return

        # Log every N batches to reduce overhead
        if not self._should_log_batch(batch_idx + 1):
            return

        total_batches = trainer.num_training_batches

        progress_percent = (
            ((batch_idx + 1) / total_batches) * 100
            if total_batches > 0
            else 0.0
        )

        epoch_duration = self._elapsed(self._train_epoch_start_time)
        batch_total = self._elapsed(self._train_batch_start_time)

        trainer.logger.log_metrics(
            {
                "train/epoch_progress_percent": progress_percent,
                "train/epoch_duration_sec": epoch_duration,
                "train/batch_total_sec": batch_total,
            },
            step=self._train_batch_step,
        )

    @rank_zero_only
    @override
    def on_train_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: "lxt.LuxonisLightningModule",
    ) -> None:
        if trainer.logger is None:
            return

        epoch_duration = self._elapsed(self._train_epoch_start_time)

        trainer.logger.log_metrics(
            {
                "train/epoch_completion_sec": epoch_duration,
                "train/epoch_progress_percent": 100.0,
            },
            step=self._train_batch_step,
        )

    @override
    def on_validation_epoch_start(
        self,
        trainer: pl.Trainer,
        pl_module: "lxt.LuxonisLightningModule",
    ) -> None:
        self._val_epoch_start_time = self._now()
        self._val_epoch_batch_count = 0

        if trainer.sanity_checking or trainer.logger is None:
            return

        trainer.logger.log_metrics(
            {"val/epoch_progress_percent": 0.0},
            step=self._val_batch_step,
        )

    @override
    def on_validation_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: "lxt.LuxonisLightningModule",
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if trainer.sanity_checking:
            return

        self._val_batch_start_time = self._now()

    @rank_zero_only
    @override
    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: "lxt.LuxonisLightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if trainer.sanity_checking:
            return

        self._val_epoch_batch_count += 1
        self._val_batch_step += 1

        if trainer.logger is None:
            return

        if not self._should_log_batch(self._val_epoch_batch_count):
            return

        total_batches = self._total_batches(trainer.num_val_batches)
        progress_percent = (
            (self._val_epoch_batch_count / total_batches) * 100
            if total_batches > 0
            else 0.0
        )
        epoch_duration = self._elapsed(self._val_epoch_start_time)

        trainer.logger.log_metrics(
            {
                "val/batch_total_sec": self._elapsed(
                    self._val_batch_start_time
                ),
                "val/epoch_progress_percent": progress_percent,
                "val/epoch_duration_sec": epoch_duration,
            },
            step=self._val_batch_step,
        )

    @rank_zero_only
    @override
    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: "lxt.LuxonisLightningModule",
    ) -> None:
        if trainer.sanity_checking or trainer.logger is None:
            return

        epoch_duration = self._elapsed(self._val_epoch_start_time)

        if self._val_epoch_batch_count > 0 and not self._should_log_batch(
            self._val_epoch_batch_count
        ):
            trainer.logger.log_metrics(
                {
                    "val/epoch_progress_percent": 100.0,
                    "val/epoch_duration_sec": epoch_duration,
                },
                step=self._val_batch_step,
            )
        trainer.logger.log_metrics(
            {"val/epoch_completion_sec": epoch_duration},
            step=trainer.current_epoch,
        )

    @override
    def on_test_epoch_start(
        self,
        trainer: pl.Trainer,
        pl_module: "lxt.LuxonisLightningModule",
    ) -> None:
        self._test_epoch_start_time = self._now()
        self._test_epoch_batch_count = 0

        if trainer.logger is None:
            return

        trainer.logger.log_metrics(
            {"test/epoch_progress_percent": 0.0},
            step=self._test_batch_step,
        )

    @override
    def on_test_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: "lxt.LuxonisLightningModule",
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._test_batch_start_time = self._now()

    @rank_zero_only
    @override
    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: "lxt.LuxonisLightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._test_epoch_batch_count += 1
        self._test_batch_step += 1

        if trainer.logger is None:
            return

        if not self._should_log_batch(self._test_epoch_batch_count):
            return

        total_batches = self._total_batches(trainer.num_test_batches)
        progress_percent = (
            (self._test_epoch_batch_count / total_batches) * 100
            if total_batches > 0
            else 0.0
        )
        epoch_duration = self._elapsed(self._test_epoch_start_time)

        trainer.logger.log_metrics(
            {
                "test/batch_total_sec": self._elapsed(
                    self._test_batch_start_time
                ),
                "test/epoch_progress_percent": progress_percent,
                "test/epoch_duration_sec": epoch_duration,
            },
            step=self._test_batch_step,
        )

    @rank_zero_only
    @override
    def on_test_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: "lxt.LuxonisLightningModule",
    ) -> None:
        if trainer.logger is None:
            return

        epoch_duration = self._elapsed(self._test_epoch_start_time)

        if self._test_epoch_batch_count > 0 and not self._should_log_batch(
            self._test_epoch_batch_count
        ):
            trainer.logger.log_metrics(
                {
                    "test/epoch_progress_percent": 100.0,
                    "test/epoch_duration_sec": epoch_duration,
                },
                step=self._test_batch_step,
            )
        trainer.logger.log_metrics(
            {"test/epoch_completion_sec": epoch_duration},
            step=trainer.current_epoch,
        )

    def _should_log_batch(self, seen_batches: int) -> bool:
        return seen_batches % self.log_every_n_batches == 0
