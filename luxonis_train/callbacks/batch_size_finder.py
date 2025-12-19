from typing import Literal

import lightning.pytorch as pl
from lightning.pytorch.callbacks import BatchSizeFinder
from loguru import logger
import luxonis_train as lxt
from typing_extensions import override


class LuxonisBatchSizeFinder(BatchSizeFinder):
    """Batch size finder callback that also updates the config batch size.
    https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.BatchSizeFinder.html
    """

    def __init__(
        self,
        mode: Literal["power", "binsearch"] = "power",  # maybe for now we force power? So no weird batch sizes like 17
        steps_per_trial: int = 3,
        init_val: int = 2,
        max_trials: int = 25,
    ):  # if we upgrade to Lightning 2.6.0, we can add the "margin" and the max_val parameters
        """
        @type mode: Literal["power", "binsearch"]
        @param mode: The mode to use for batch size scaling.
            "power" - Increases batch size by powers of 2 until OOM.
            "binsearch" - Binary search between last successful and
            OOM batch sizes.
        @type steps_per_trial: int
        @param steps_per_trial: Number of steps to run for each batch
            size trial.
        @type init_val: int
        @param init_val: Initial batch size to start the search from.
        @type max_trials: int
        @param max_trials: Maximum number of trials to run.
        """
        super().__init__(
            mode=mode,
            steps_per_trial=steps_per_trial,
            init_val=init_val,
            max_trials=max_trials,
            batch_arg_name="batch_size",
        )
        self._original_batch_size: int | None = None

    @override
    def on_fit_start(
        self, trainer: pl.Trainer, pl_module: "lxt.LuxonisLightningModule"
    ) -> None:
        """
        Stores the original batch size, runs the batch size finder
        and updates the config with the optimal batch size.
        """
        self._original_batch_size = pl_module.cfg.trainer.batch_size

        super().on_fit_start(trainer, pl_module)  # sets self.optimal_batch_size

        self._update_config_batch_size(pl_module)

    def _update_config_batch_size(
        self, pl_module: "lxt.LuxonisLightningModule"
    ) -> None:
        """Updates the config batch size with the optimal batch size.

        Also logs appropriate messages about the batch size change.

        @type pl_module: L{pl.LightningModule}
        @param pl_module: PyTorch Lightning module.
        """
        if self.optimal_batch_size is None:
            return

        if self._original_batch_size is not None:
            if self.optimal_batch_size > self._original_batch_size:
                logger.warning(
                    f"LuxonisBatchSizeFinder increased batch size from ({self._original_batch_size}) "
                    f"higher than the configured batch size ({self.optimal_batch_size}). "
                    "This suggests your hardware can handle larger batches. "
                )
            elif self.optimal_batch_size < self._original_batch_size:
                logger.info(
                    f"LuxonisBatchSizeFinder reduced batch size from {self._original_batch_size} "
                    f"to {self.optimal_batch_size} to avoid OOM problems."
                )
            else:
                logger.info(
                    f"LuxonisBatchSizeFinder confirmed batch size {self.optimal_batch_size} "
                    "is optimal."
                )

        pl_module.cfg.trainer.batch_size = self.optimal_batch_size
        logger.info(
            f"Config batch size updated to {self.optimal_batch_size}."
        )
