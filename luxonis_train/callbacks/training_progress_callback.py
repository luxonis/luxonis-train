"""Logs the progress and the timing of each batch and each epoch."""

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
    """Callback that logs the progress and the timing of each loop.

    The callback sends these keys to ``trainer.logger`` with
    ``log_metrics``, where ``<mode>`` is ``train``, ``val``, or ``test``:

    - ``<mode>/epoch_progress_percent``: The share of the batches of the
      epoch that are done, in percent.
    - ``<mode>/epoch_duration_sec``: The seconds from the start of the
      epoch to the log entry.
    - ``<mode>/batch_total_sec``: The seconds from the start hook to the
      end hook of the logged batch.
    - ``<mode>/epoch_completion_sec``: The seconds from the start to the
      end of the epoch.

    The constructor sets one batch counter for each loop to 0. A counter
    does not reset between epochs. The counter of a loop is the step of
    each key of that loop. ``val/epoch_completion_sec`` and
    ``test/epoch_completion_sec`` are the exceptions. Their step is
    ``trainer.current_epoch``.

    A batch-end hook logs only when the number of batches done in the
    epoch is a multiple of ``log_every_n_batches``. The callback ignores
    the validation sanity check. The batch-end and epoch-end hooks run
    on rank zero only. Without ``trainer.logger``, the callback sends no
    keys. The start of each train epoch then logs a warning.
    `LuxonisLightningModule.get_mlflow_logging_keys` lists the twelve
    keys when the config lists this callback.

    Add the callback to ``trainer.callbacks`` of the config:

    .. code-block:: yaml

        trainer:
          callbacks:
            - name: TrainingProgressCallback
              params:
                log_every_n_batches: 10

    Attributes:
        log_every_n_batches (int): The interval of the batch-end logs,
            in batches. At least ``1``.

    """

    def __init__(self, log_every_n_batches: int = 1):
        """Initialize the callback.

        Args:
            log_every_n_batches (int): Log at the end of a batch once
                every this many batches of an epoch. ``1`` logs every
                batch. A higher value reduces the logging overhead. A
                value below ``1`` acts as ``1``.

        Example:
            >>> from luxonis_train.callbacks import TrainingProgressCallback
            >>> callback = TrainingProgressCallback(log_every_n_batches=0)
            >>> callback.log_every_n_batches
            1

        """
        super().__init__()
        self._log_every_n_batches = max(1, log_every_n_batches)
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
        """Sum the finite batch counts of the evaluation data loaders.

        Args:
            total_batches (float | list[int | float]): The batch count of
                one data loader, or a list with one count for each.

        Returns:
            int: The sum of the finite counts. A single infinite count
            gives ``0``.

        Example:
            >>> from luxonis_train.callbacks import TrainingProgressCallback
            >>> TrainingProgressCallback._total_batches([3, float("inf"), 2])
            5
            >>> TrainingProgressCallback._total_batches(float("inf"))
            0

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
        """Start the timer of the train epoch and log zero progress.

        Lightning calls this hook at the start of every training epoch.
        The hook stores the start time. It then logs
        ``train/epoch_progress_percent`` as ``0.0`` at the step of the
        train batch counter. Without ``trainer.logger``, it logs a
        warning instead.

        Args:
            trainer (``pl.Trainer``): The trainer. The hook logs to its
                ``logger``.
            pl_module (LuxonisLightningModule): The model. Unused.

        """
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
        """Start the timer of the train batch.

        Lightning calls this hook before the training step of every
        batch. The hook stores the start time of
        ``train/batch_total_sec``.

        Args:
            trainer (``pl.Trainer``): The trainer. Unused.
            pl_module (LuxonisLightningModule): The model. Unused.
            batch (``Any``): The batch. Unused.
            batch_idx (int): The index of the batch in the epoch. Unused.

        """
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
        """Log the progress and the timing of the train batch.

        Lightning calls this hook at the end of every training batch.
        The hook runs on rank zero only. It adds 1 to the train batch
        counter. When ``batch_idx + 1`` is a multiple of
        ``log_every_n_batches``, it logs these keys at the step of that
        counter:

        - ``train/epoch_progress_percent``: :math:`100 (i + 1) / n`,
          where :math:`i` is ``batch_idx`` and :math:`n` is
          ``trainer.num_training_batches``. The value is ``0.0`` when
          :math:`n` is ``0`` or infinite.
        - ``train/epoch_duration_sec``: The seconds since
          `on_train_epoch_start`.
        - ``train/batch_total_sec``: The seconds since
          `on_train_batch_start`.

        Without ``trainer.logger``, the hook logs nothing.

        Args:
            trainer (``pl.Trainer``): The trainer. The hook reads
                ``num_training_batches`` and logs to ``logger``.
            pl_module (LuxonisLightningModule): The model. Unused.
            outputs (``STEP_OUTPUT``): The output of the training step.
                Unused.
            batch (``Any``): The batch. Unused.
            batch_idx (int): The index of the batch in the epoch.

        """
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
        """Log the duration of the finished train epoch.

        Lightning calls this hook at the end of every training epoch.
        When a validation runs in the epoch, the call comes after it, so
        the duration includes the validation time. The hook runs on rank
        zero only. It logs ``train/epoch_completion_sec``, the seconds
        since `on_train_epoch_start`, and ``train/epoch_progress_percent``
        as ``100.0``. Both keys use the step of the train batch counter.
        Without ``trainer.logger``, the hook logs nothing.

        Args:
            trainer (``pl.Trainer``): The trainer. The hook logs to its
                ``logger``.
            pl_module (LuxonisLightningModule): The model. Unused.

        """
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
        """Start the validation epoch timer and log zero progress.

        Lightning calls this hook at the start of every validation epoch,
        the sanity check included. The hook stores the start time and
        sets the batch count of the epoch to 0. Outside the sanity check,
        it logs ``val/epoch_progress_percent`` as ``0.0`` at the step of
        the validation batch counter. Without ``trainer.logger``, it logs
        nothing.

        Args:
            trainer (``pl.Trainer``): The trainer. The hook reads
                ``sanity_checking`` and logs to ``logger``.
            pl_module (LuxonisLightningModule): The model. Unused.

        """
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
        """Start the timer of the validation batch.

        Lightning calls this hook before the validation step of every
        batch. Outside the sanity check, the hook stores the start time
        of ``val/batch_total_sec``.

        Args:
            trainer (``pl.Trainer``): The trainer. The hook reads
                ``sanity_checking``.
            pl_module (LuxonisLightningModule): The model. Unused.
            batch (``Any``): The batch. Unused.
            batch_idx (int): The index of the batch. Unused.
            dataloader_idx (int): The index of the data loader. Unused.

        """
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
        """Log the progress and the timing of the validation batch.

        Lightning calls this hook at the end of every validation batch.
        The hook runs on rank zero only and does nothing during the
        sanity check. It adds 1 to the batch count of the epoch and to
        the validation batch counter. When the batch count is a multiple
        of ``log_every_n_batches``, it logs these keys at the step of the
        validation batch counter:

        - ``val/batch_total_sec``: The seconds since
          `on_validation_batch_start`.
        - ``val/epoch_progress_percent``: :math:`100 c / n`, where
          :math:`c` is the batch count and :math:`n` is the sum of the
          finite entries of ``trainer.num_val_batches``. The value is
          ``0.0`` when :math:`n` is ``0``.
        - ``val/epoch_duration_sec``: The seconds since
          `on_validation_epoch_start`.

        The batch count covers the batches of every validation data
        loader. Without ``trainer.logger``, the hook logs nothing.

        Args:
            trainer (``pl.Trainer``): The trainer. The hook reads
                ``sanity_checking`` and ``num_val_batches``, and logs to
                ``logger``.
            pl_module (LuxonisLightningModule): The model. Unused.
            outputs (``STEP_OUTPUT``): The output of the validation step.
                Unused.
            batch (``Any``): The batch. Unused.
            batch_idx (int): The index of the batch. Unused.
            dataloader_idx (int): The index of the data loader. Unused.

        """
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
        """Log the duration of the finished validation epoch.

        Lightning calls this hook at the end of every validation epoch.
        The hook runs on rank zero only. It does nothing during the
        sanity check or without ``trainer.logger``.

        When the batch count of the epoch is above 0 and not a multiple
        of ``log_every_n_batches``, the last batch has no log entry. The
        hook then logs ``val/epoch_progress_percent`` as ``100.0`` and
        ``val/epoch_duration_sec`` at the step of the validation batch
        counter. In every case, it logs ``val/epoch_completion_sec``, the
        seconds since `on_validation_epoch_start`, at the step
        ``trainer.current_epoch``.

        Args:
            trainer (``pl.Trainer``): The trainer. The hook reads
                ``sanity_checking`` and ``current_epoch``, and logs to
                ``logger``.
            pl_module (LuxonisLightningModule): The model. Unused.

        """
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
        """Start the timer of the test epoch and log zero progress.

        Lightning calls this hook at the start of every test epoch. The
        hook stores the start time and sets the batch count of the epoch
        to 0. It then logs ``test/epoch_progress_percent`` as ``0.0`` at
        the step of the test batch counter. Without ``trainer.logger``, it
        logs nothing.

        Args:
            trainer (``pl.Trainer``): The trainer. The hook logs to its
                ``logger``.
            pl_module (LuxonisLightningModule): The model. Unused.

        """
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
        """Start the timer of the test batch.

        Lightning calls this hook before the test step of every batch.
        The hook stores the start time of ``test/batch_total_sec``.

        Args:
            trainer (``pl.Trainer``): The trainer. Unused.
            pl_module (LuxonisLightningModule): The model. Unused.
            batch (``Any``): The batch. Unused.
            batch_idx (int): The index of the batch. Unused.
            dataloader_idx (int): The index of the data loader. Unused.

        """
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
        """Log the progress and the timing of the test batch.

        Lightning calls this hook at the end of every test batch. The
        hook runs on rank zero only. It adds 1 to the batch count of the
        epoch and to the test batch counter. When the batch count is a
        multiple of ``log_every_n_batches``, it logs these keys at the
        step of the test batch counter:

        - ``test/batch_total_sec``: The seconds since
          `on_test_batch_start`.
        - ``test/epoch_progress_percent``: :math:`100 c / n`, where
          :math:`c` is the batch count and :math:`n` is the sum of the
          finite entries of ``trainer.num_test_batches``. The value is
          ``0.0`` when :math:`n` is ``0``.
        - ``test/epoch_duration_sec``: The seconds since
          `on_test_epoch_start`.

        The batch count covers the batches of every test data loader.
        Without ``trainer.logger``, the hook logs nothing.

        Args:
            trainer (``pl.Trainer``): The trainer. The hook reads
                ``num_test_batches`` and logs to ``logger``.
            pl_module (LuxonisLightningModule): The model. Unused.
            outputs (``STEP_OUTPUT``): The output of the test step.
                Unused.
            batch (``Any``): The batch. Unused.
            batch_idx (int): The index of the batch. Unused.
            dataloader_idx (int): The index of the data loader. Unused.

        """
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
        """Log the duration of the finished test epoch.

        Lightning calls this hook at the end of every test epoch. The
        hook runs on rank zero only. It does nothing without
        ``trainer.logger``.

        When the batch count of the epoch is above 0 and not a multiple
        of ``log_every_n_batches``, the last batch has no log entry. The
        hook then logs ``test/epoch_progress_percent`` as ``100.0`` and
        ``test/epoch_duration_sec`` at the step of the test batch
        counter. In every case, it logs ``test/epoch_completion_sec``,
        the seconds since `on_test_epoch_start`, at the step
        ``trainer.current_epoch``.

        Args:
            trainer (``pl.Trainer``): The trainer. The hook reads
                ``current_epoch`` and logs to ``logger``.
            pl_module (LuxonisLightningModule): The model. Unused.

        """
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
        return seen_batches % self._log_every_n_batches == 0
