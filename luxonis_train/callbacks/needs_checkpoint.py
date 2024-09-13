import logging
from typing import Literal

import lightning.pytorch as pl

import luxonis_train

logger = logging.getLogger(__name__)


class NeedsCheckpoint(pl.Callback):
    def __init__(
        self,
        preferred_checkpoint: Literal["metric", "loss"] = "metric",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.preferred_checkpoint = preferred_checkpoint

    @staticmethod
    def _get_checkpoint(
        checkpoint_type: str,
        pl_module: "luxonis_train.models.LuxonisLightningModule",
    ) -> str | None:
        if checkpoint_type == "loss":
            path = pl_module.core.get_min_loss_checkpoint_path()
            if not path:
                logger.error(
                    "No checkpoint for minimum loss found. "
                    "Make sure that `ModelCheckpoint` callback is present "
                    "and at least one validation epoch has been performed."
                )
            return path
        else:
            path = pl_module.core.get_best_metric_checkpoint_path()
            if not path:
                logger.error(
                    "No checkpoint for best metric found. "
                    "Make sure that `ModelCheckpoint` callback is present, "
                    "at least one validation epoch has been performed and "
                    "the model has at least one metric."
                )
            return path

    @staticmethod
    def _get_other_type(checkpoint_type: str) -> str:
        if checkpoint_type == "loss":
            return "metric"
        return "loss"

    def get_checkpoint(
        self, pl_module: "luxonis_train.models.LuxonisLightningModule"
    ) -> str | None:
        path = self._get_checkpoint(self.preferred_checkpoint, pl_module)
        if path is not None:
            return path
        other_checkpoint = self._get_other_type(self.preferred_checkpoint)
        logger.info(f"Attempting to use {other_checkpoint} checkpoint.")
        return self._get_checkpoint(other_checkpoint, pl_module)
