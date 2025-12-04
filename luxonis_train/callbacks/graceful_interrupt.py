import os
import signal
import sys

import lightning.pytorch as pl
from loguru import logger
from pathlib import Path


class GracefulInterruptCallback(pl.Callback):
    """
    Handles SIGINT/SIGTERM:
      - First interrupt: save checkpoint, stop training, skip all train-end callbacks
      - Second interrupt: immediate exit, skip saving resume.ckpt
    """

    def __init__(self, save_dir: Path, tracker):
        super().__init__()
        self.save_dir = Path(save_dir)
        self.tracker = tracker
        self._interrupted_once = False
        self._interrupted = False
        self._main_pid = os.getpid()

    def setup(self, trainer, pl_module, stage=None):
        self._trainer = trainer

        if os.getpid() == self._main_pid:  # to avoid problem with multiple workers
            signal.signal(signal.SIGINT, self._handle_signal)
            signal.signal(signal.SIGTERM, self._handle_signal)

        logger.info("Added GracefulInterrupt callback")

    def _handle_signal(self, signum, frame):
        if self._interrupted_once:
            logger.warning("Second interrupt, forcing immediate exit.")
            os._exit(1)

        self._interrupted_once = True
        self._interrupted = True

        self._save_interrupt_checkpoint()

        if self._trainer:
            self._trainer.should_stop = True

    def _save_interrupt_checkpoint(self):
        ckpt_path = self.save_dir / "resume.ckpt"
        logger.warning(f"Saving interrupt checkpoint to: {ckpt_path}")

        try:
            self._trainer.save_checkpoint(ckpt_path)
        except Exception:
            logger.exception("Failed to save interrupt checkpoint.")

        try:
            self.tracker.upload_artifact(ckpt_path, typ="checkpoints", name="resume.ckpt")
            self.tracker._finalize(status="failed")
        except Exception:
            logger.exception("Failed to upload checkpoint or finalize tracker.")

    def on_train_end(self, trainer, pl_module):
        """
        Prevent all other train-end callbacks (TestOnTrainEnd, ExportOnTrainEnd, etc)
        from running if the training terminated due to interrupt.
        """

        if not self._interrupted:
            return

        logger.warning("Graceful shutdown, skipping all remaining train-end callbacks.")
        sys.exit(0)
