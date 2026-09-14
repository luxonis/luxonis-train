"""Turns the first interrupt into a clean stop.

The callback saves a resume checkpoint and skips the remaining train-end
callbacks. A second interrupt exits at once.

"""

import os
import signal
import sys
from pathlib import Path
from types import FrameType
from typing import Any

import lightning.pytorch as pl
from loguru import logger

import luxonis_train as lxt
from luxonis_train.utils.tracker import LuxonisTrackerPL


class GracefulInterruptCallback(pl.Callback):
    """Callback that stops a fit cleanly on ``SIGINT`` or ``SIGTERM``.

    `LuxonisModel` adds this callback to its main trainer and to the
    trainer of each tuning trial. While a fit runs, the callback
    replaces the handlers of ``SIGINT`` and ``SIGTERM``. The new handler
    acts as follows:

    - **First signal**: the handler logs a warning with the path and
      saves ``resume.ckpt`` in ``save_dir``. When the callback has a
      tracker, the handler then uploads the checkpoint and finalizes
      the run with the status ``"failed"``. The handler logs the error
      of a failed step and does not raise it. A failed save does not
      stop the upload. A failed upload skips the finalization. Then the
      handler sets ``trainer.should_stop`` to ``True``.
    - **Second signal**: the handler logs a warning and calls
      ``os._exit(1)``. The process ends at once, without cleanup.

    After the first ``SIGINT``, Lightning ends the fit. Then
    `GracefulInterruptCallback.on_train_end` raises ``SystemExit``, so
    no train-end hook of a later callback runs. When the fit starts to
    run, Lightning installs its own ``SIGTERM`` handler, which also
    calls the handler of the callback. After the first ``SIGTERM``,
    Lightning raises its ``SIGTERMException``, a ``SystemExit``, at the
    end of the current batch or epoch. Then no train-end hook runs.

    The handler ignores a signal in a process other than the one that
    created the callback, for example in a data loader worker.

    """

    def __init__(
        self, save_dir: Path, tracker: LuxonisTrackerPL | None = None
    ):
        """Initialize the callback.

        Args:
            save_dir (``Path``): The directory for ``resume.ckpt``. The
                callback converts the value to a `pathlib.Path`, so a
                ``str`` is also valid.
            tracker (LuxonisTrackerPL | None): The tracker that receives
                ``resume.ckpt`` on the first interrupt. The first
                interrupt also finalizes its run with the status
                ``"failed"``. ``None`` skips the upload and the
                finalization.

        """
        super().__init__()
        self.save_dir = Path(save_dir)
        self.tracker = tracker
        self._interrupted_once = False
        self._interrupted = False
        self._trainer: pl.Trainer | None = None
        self._main_pid = os.getpid()
        self._signal_handlers: dict[int, Any] = {}

    def setup(
        self,
        trainer: pl.Trainer,
        pl_module: "lxt.LuxonisLightningModule",
        stage: str | None = None,
    ) -> None:
        """Install the signal handler when a fit starts.

        Lightning calls this hook at the start of every stage. The hook
        stores ``trainer`` for every stage, so that the handler can save
        a checkpoint and stop the trainer. For a stage other than
        ``"fit"``, the hook does nothing else.

        For ``"fit"``, in the process that created the callback, the
        hook saves the current handlers of ``SIGINT`` and ``SIGTERM``
        and installs its own handler. In any process, the hook then
        logs ``Added GracefulInterrupt callback`` at the ``INFO`` level.

        Args:
            trainer (``pl.Trainer``): The trainer to save and stop on an
                interrupt.
            pl_module (LuxonisLightningModule): The model. Unused.
            stage (str | None): The stage that starts, for example
                ``"fit"``.

        """
        self._trainer = trainer

        if stage != "fit":
            return

        if os.getpid() == self._main_pid:
            for signum in (signal.SIGINT, signal.SIGTERM):
                self._signal_handlers[signum] = signal.getsignal(signum)
                signal.signal(signum, self._handle_signal)

        logger.info("Added GracefulInterrupt callback")

    def teardown(
        self,
        trainer: pl.Trainer,
        pl_module: "lxt.LuxonisLightningModule",
        stage: str | None = None,
    ) -> None:
        """Restore the signal handlers when a fit ends.

        Lightning calls this hook at the end of every stage that
        finishes without an exception. For ``"fit"``, in the process
        that created the callback, the hook puts back the handlers that
        `GracefulInterruptCallback.setup` saved. For other stages and
        in other processes, the hook does nothing. After an interrupt,
        Lightning does not call this hook, so the handler of the
        callback stays installed.

        Args:
            trainer (``pl.Trainer``): The trainer. Unused.
            pl_module (LuxonisLightningModule): The model. Unused.
            stage (str | None): The stage that ends.

        """
        if stage != "fit":
            return

        if os.getpid() == self._main_pid:
            for signum, handler in self._signal_handlers.items():
                signal.signal(signum, handler)
            self._signal_handlers.clear()

    def _handle_signal(self, signum: int, frame: FrameType | None) -> None:
        if os.getpid() != self._main_pid:
            return

        if self._interrupted_once:
            logger.warning("Second interrupt, forcing immediate exit.")
            os._exit(1)

        self._interrupted_once = True
        self._interrupted = True

        self._save_interrupt_checkpoint()

        if self._trainer:
            self._trainer.should_stop = True

    def _save_interrupt_checkpoint(self) -> None:
        ckpt_path = self.save_dir / "resume.ckpt"
        logger.warning(f"Saving interrupt checkpoint to: {ckpt_path}")

        if self._trainer is None:
            logger.error(
                "Trainer not yet set, cannot save interrupt checkpoint."
            )
            return

        try:
            self._trainer.save_checkpoint(ckpt_path)
        except Exception:
            logger.exception("Failed to save interrupt checkpoint.")

        try:
            if self.tracker:
                self.tracker.upload_artifact(
                    ckpt_path, typ="checkpoints", name="resume.ckpt"
                )
                self.tracker._finalize(status="failed")
        except Exception:
            logger.exception(
                "Failed to upload checkpoint or finalize tracker."
            )

    def on_train_end(
        self, trainer: pl.Trainer, pl_module: "lxt.LuxonisLightningModule"
    ) -> None:
        """Stop the process after an interrupted fit.

        Lightning calls this hook at the end of a fit. After a fit
        without an interrupt, the hook does nothing. After an interrupt,
        the hook logs a warning and calls ``sys.exit(0)``. After a
        ``SIGTERM``, Lightning raises its own exception before this hook
        runs.

        The ``SystemExit`` stops the train-end hooks of the callbacks
        that come after this one. `LuxonisModel` gives this callback to
        the trainer. Lightning adds the callbacks of
        `LuxonisLightningModule.configure_callbacks` after it. Thus the
        callbacks of the config, for example `TestOnTrainEnd` and
        `ExportOnTrainEnd`, come later. When the config lists this
        callback itself, Lightning drops the instance that
        `LuxonisModel` created. The instance of the config then runs at
        its position in the config. After the exception, Lightning
        calls the ``on_exception`` hooks, but not the ``teardown``
        hooks.

        Args:
            trainer (``pl.Trainer``): The trainer. Unused.
            pl_module (LuxonisLightningModule): The model. Unused.

        Raises:
            SystemExit: With the code ``0``, when an interrupt stopped
                the fit.

        """
        if not self._interrupted:
            return

        logger.warning(
            "Graceful shutdown, skipping all remaining train-end callbacks."
        )
        sys.exit(0)
