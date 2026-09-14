"""The base class of the callbacks that act on the best checkpoint.

`NeedsCheckpoint.get_checkpoint` returns the path of the checkpoint with
the best main metric or with the lowest validation loss. When the
preferred checkpoint does not exist, it tries the other one. When
neither exists, it returns ``None``, and each subclass decides what to
do.

"""

from typing import Literal

import lightning.pytorch as pl
from loguru import logger

import luxonis_train as lxt


class NeedsCheckpoint(pl.Callback):
    """Base class of the callbacks that act on the best checkpoint.

    `ArchiveOnTrainEnd`, `ConvertOnTrainEnd`, `ExportOnTrainEnd`,
    `AIMETCallback`, and `TestOnTrainEnd` inherit from it. Each one calls
    `get_checkpoint` in its ``on_train_end`` hook. `ArchiveOnTrainEnd`
    calls it only when no earlier ONNX export exists. This class defines
    no hook of its own.

    Attributes:
        preferred_checkpoint (``Literal["metric", "loss"]``): The
            checkpoint that `get_checkpoint` tries first. ``"metric"``
            selects the best main metric, and ``"loss"`` selects the
            lowest validation loss.

    """

    def __init__(
        self,
        preferred_checkpoint: Literal["metric", "loss"] = "metric",
        **kwargs,
    ):
        """Initialize the callback.

        Args:
            preferred_checkpoint (``Literal["metric", "loss"]``): The
                checkpoint to try first. ``"metric"`` selects the best
                main metric, and ``"loss"`` selects the lowest validation
                loss. The constructor does not check the value. Any value
                other than ``"loss"`` acts as ``"metric"``.
            **kwargs (``Any``): Keyword arguments for ``pl.Callback``.
                That class accepts none, so any key raises
                ``TypeError``.

        """
        super().__init__(**kwargs)
        self.preferred_checkpoint = preferred_checkpoint

    @staticmethod
    def _get_checkpoint(
        checkpoint_type: str, pl_module: "lxt.LuxonisLightningModule"
    ) -> str | None:
        if checkpoint_type == "loss":
            path = pl_module.core.get_min_loss_checkpoint_path()
            if not path:
                logger.error(
                    "No checkpoint for minimum loss found. "
                    "Make sure that `ModelCheckpoint` callback is present "
                    "and at least one validation epoch has been performed."
                )
                return None
            return path
        path = pl_module.core.get_best_metric_checkpoint_path()
        if not path:
            logger.error(
                "No checkpoint for best metric found. "
                "Make sure that `ModelCheckpoint` callback is present, "
                "at least one validation epoch has been performed and "
                "the model has at least one metric."
            )
            return None
        return path

    @staticmethod
    def _get_other_type(checkpoint_type: str) -> str:
        if checkpoint_type == "loss":
            return "metric"
        return "loss"

    def get_checkpoint(
        self, pl_module: "lxt.LuxonisLightningModule"
    ) -> str | None:
        """Return the path of the best checkpoint.

        The method reads the ``best_model_path`` of a ``ModelCheckpoint``
        callback through ``pl_module.core``. It calls
        `LuxonisModel.get_best_metric_checkpoint_path` for ``"metric"``
        and `LuxonisModel.get_min_loss_checkpoint_path` for ``"loss"``.
        The metric checkpoint exists only when the config has a metric.
        Without an ``is_main_metric`` flag, the config makes the first
        metric the main metric. A path stays empty until its callback
        saves a checkpoint.

        The method tries ``preferred_checkpoint`` first. When that path
        is ``None`` or empty, it logs an error and an info message, and
        tries the other checkpoint. It logs a second error when that
        path is missing too.

        Both getters run on rank zero only and return ``None`` on every
        other rank. There, the method logs the same three messages and
        returns ``None``.

        Args:
            pl_module (LuxonisLightningModule): The module of the run.
                Its `LuxonisModel` holds the trainer and its callbacks.

        Returns:
            str | None: The path of the checkpoint file, or ``None`` when
            neither checkpoint exists.

        """
        path = self._get_checkpoint(self.preferred_checkpoint, pl_module)
        if path is not None:
            return path
        other_checkpoint = self._get_other_type(self.preferred_checkpoint)
        logger.info(f"Attempting to use {other_checkpoint} checkpoint.")
        return self._get_checkpoint(other_checkpoint, pl_module)
