import os
import signal
import sys
from pathlib import Path
from types import FrameType

import lightning.pytorch as pl
from loguru import logger

import luxonis_train as lxt
from luxonis_train.utils.tracker import LuxonisTrackerPL


class GracefulInterruptCallback(pl.Callback):
    """Handles SIGINT/SIGTERM.

    Behavior:
     - First interrupt: save checkpoint, stop training, skip all train-end callbacks.
     - Second interrupt: immediate exit, skip saving resume.ckpt.
    """

    def __init__(
        self, save_dir: Path, tracker: LuxonisTrackerPL | None = None
    ):
        super().__init__()
        self.save_dir = Path(save_dir)
        self.tracker = tracker
        self._interrupted_once = False
        self._interrupted = False
        self._trainer: pl.Trainer | None = None
        self._main_pid = os.getpid()

    def setup(
        self,
        trainer: pl.Trainer,
        pl_module: "lxt.LuxonisLightningModule",
        stage: str | None = None,
    ) -> None:
        self._trainer = trainer

        if (
            os.getpid() == self._main_pid
        ):  # to avoid problem with multiple workers
            signal.signal(signal.SIGINT, self._handle_signal)
            signal.signal(signal.SIGTERM, self._handle_signal)

        logger.info("Added GracefulInterrupt callback")

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
        """Prevent all other train-end callbacks (TestOnTrainEnd,
        ExportOnTrainEnd, etc) from running if the training terminated
        due to interrupt."""

        if not self._interrupted:
            return

        logger.warning(
            "Graceful shutdown, skipping all remaining train-end callbacks."
        )
        sys.exit(0)
