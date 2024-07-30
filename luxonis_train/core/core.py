import os
import os.path as osp
from logging import getLogger
from typing import Any

import lightning.pytorch as pl
import lightning_utilities.core.rank_zero as rank_zero_module
import rich.traceback
import torch
import torch.utils.data as torch_data
from lightning.pytorch.utilities import rank_zero_only  # type: ignore
from luxonis_ml.data import Augmentations
from luxonis_ml.utils import reset_logging, setup_logging

from luxonis_train.callbacks import LuxonisProgressBar
from luxonis_train.utils.config import Config
from luxonis_train.utils.general import DatasetMetadata
from luxonis_train.utils.loaders import collate_fn
from luxonis_train.utils.registry import LOADERS
from luxonis_train.utils.tracker import LuxonisTrackerPL

logger = getLogger(__name__)


class Core:
    """Common logic of the core components.

    This class contains common logic of the core components (trainer, evaluator,
    exporter, etc.).
    """

    def __init__(
        self,
        cfg: str | dict[str, Any] | Config | None,
        opts: list[str] | tuple[str, ...] | dict[str, Any] | None = None,
    ):
        """Constructs a new Core instance.

        Loads the config and initializes datasets, dataloaders, augmentations,
        lightning components, etc.

        @type cfg: str | dict[str, Any] | Config
        @param cfg: Path to config file or config dict used to setup training

        @type opts: list[str] | tuple[str, ...] | dict[str, Any] | None
        @param opts: Argument dict provided through command line, used for config overriding
        """

        if isinstance(cfg, Config):
            self.cfg = cfg
        else:
            self.cfg = Config.get_config(cfg, opts)

        opts = opts or []

        if self.cfg.use_rich_text:
            rich.traceback.install(suppress=[pl, torch], show_locals=False)

        self.rank = rank_zero_only.rank

        self.tracker = self._create_tracker()
        # NOTE: tracker.experiment has to be called first in order
        # for the run_id to be initialized
        # TODO: it shouldn't be a property because of the above
        # _ = self.tracker.experiment
        self._run_id = self.tracker.run_id

        self.run_save_dir = os.path.join(
            self.cfg.tracker.save_directory, self.tracker.run_name
        )
        self.log_file = osp.join(self.run_save_dir, "luxonis_train.log")

        # NOTE: to add the file handler (we only get the save dir now,
        # but we want to use the logger before)
        reset_logging()
        setup_logging(
            use_rich=self.cfg.use_rich_text,
            file=self.log_file,
        )

        # NOTE: overriding logger in pl so it uses our logger to log device info
        rank_zero_module.log = logger

        deterministic = False
        if self.cfg.trainer.seed is not None:
            pl.seed_everything(self.cfg.trainer.seed, workers=True)
            deterministic = True

        self.train_augmentations = Augmentations(
            image_size=self.cfg.trainer.preprocessing.train_image_size,
            augmentations=[
                i.model_dump()
                for i in self.cfg.trainer.preprocessing.get_active_augmentations()
            ],
            train_rgb=self.cfg.trainer.preprocessing.train_rgb,
            keep_aspect_ratio=self.cfg.trainer.preprocessing.keep_aspect_ratio,
        )
        self.val_augmentations = Augmentations(
            image_size=self.cfg.trainer.preprocessing.train_image_size,
            augmentations=[
                i.model_dump()
                for i in self.cfg.trainer.preprocessing.get_active_augmentations()
            ],
            train_rgb=self.cfg.trainer.preprocessing.train_rgb,
            keep_aspect_ratio=self.cfg.trainer.preprocessing.keep_aspect_ratio,
            only_normalize=True,
        )

        self.pl_trainer = pl.Trainer(
            accelerator=self.cfg.trainer.accelerator,
            devices=self.cfg.trainer.devices,
            strategy=self.cfg.trainer.strategy,
            logger=self.tracker,  # type: ignore
            max_epochs=self.cfg.trainer.epochs,
            accumulate_grad_batches=self.cfg.trainer.accumulate_grad_batches,
            check_val_every_n_epoch=self.cfg.trainer.validation_interval,
            num_sanity_val_steps=self.cfg.trainer.num_sanity_val_steps,
            profiler=self.cfg.trainer.profiler,  # for debugging purposes,
            # NOTE: this is likely PL bug,
            # should be configurable inside configure_callbacks(),
            callbacks=LuxonisProgressBar() if self.cfg.use_rich_text else None,
            deterministic=deterministic,
        )

        self.loaders = {
            view: LOADERS.get(self.cfg.loader.name)(
                augmentations=(
                    self.train_augmentations
                    if view == "train"
                    else self.val_augmentations
                ),
                view={
                    "train": self.cfg.loader.train_view,
                    "val": self.cfg.loader.val_view,
                    "test": self.cfg.loader.test_view,
                }[view],
                image_source=self.cfg.loader.image_source,
                **self.cfg.loader.params,
            )
            for view in ["train", "val", "test"]
        }
        sampler = None
        if self.cfg.trainer.use_weighted_sampler:
            classes_count = self.loaders["train"].get_classes()[1]
            if len(classes_count) == 0:
                logger.warning(
                    "WeightedRandomSampler only available for classification tasks. Using default sampler instead."
                )
            else:
                weights = [1 / i for i in classes_count.values()]
                num_samples = sum(classes_count.values())
                sampler = torch_data.WeightedRandomSampler(weights, num_samples)

        self.pytorch_loaders = {
            view: torch_data.DataLoader(
                self.loaders[view],
                batch_size=self.cfg.trainer.batch_size,
                num_workers=self.cfg.trainer.num_workers,
                collate_fn=collate_fn,
                shuffle=view == "train",
                drop_last=(
                    self.cfg.trainer.skip_last_batch if view == "train" else False
                ),
                sampler=sampler if view == "train" else None,
            )
            for view in ["train", "val", "test"]
        }
        self.error_message = None

        self.dataset_metadata = DatasetMetadata.from_loader(self.loaders["train"])
        self.dataset_metadata.set_loader(self.pytorch_loaders["train"])

        self.cfg.save_data(os.path.join(self.run_save_dir, "config.yaml"))

    def set_train_augmentations(self, aug: Augmentations) -> None:
        """Sets augmentations used for training dataset."""
        self.train_augmentations = aug

    def set_val_augmentations(self, aug: Augmentations) -> None:
        """Sets augmentations used for validation dataset."""
        self.val_augmentations = aug

    def set_test_augmentations(self, aug: Augmentations) -> None:
        """Sets augmentations used for test dataset."""
        self.test_augmentations = aug

    @rank_zero_only
    def get_save_dir(self) -> str:
        """Return path to directory where checkpoints are saved.

        @rtype: str
        @return: Save directory path
        """
        return self.run_save_dir

    @rank_zero_only
    def get_error_message(self) -> str | None:
        """Return error message if one occurs while running in thread, otherwise None.

        @rtype: str | None
        @return: Error message
        """
        return self.error_message

    @rank_zero_only
    def get_min_loss_checkpoint_path(self) -> str:
        """Return best checkpoint path with respect to minimal validation loss.

        @rtype: str
        @return: Path to best checkpoint with respect to minimal validation loss
        """
        return self.pl_trainer.checkpoint_callbacks[0].best_model_path  # type: ignore

    @rank_zero_only
    def get_best_metric_checkpoint_path(self) -> str:
        """Return best checkpoint path with respect to best validation metric.

        @rtype: str
        @return: Path to best checkpoint with respect to best validation metric
        """
        return self.pl_trainer.checkpoint_callbacks[1].best_model_path  # type: ignore

    def reset_logging(self) -> None:
        """Close file handlers to release the log file."""
        reset_logging()

    def _create_tracker(self, run_id: str | None = None) -> LuxonisTrackerPL:
        kwargs = self.cfg.tracker.model_dump()
        if run_id is not None:
            kwargs["run_id"] = run_id
        return LuxonisTrackerPL(
            rank=self.rank,
            mlflow_tracking_uri=self.cfg.ENVIRON.MLFLOW_TRACKING_URI,
            **kwargs,
        )
