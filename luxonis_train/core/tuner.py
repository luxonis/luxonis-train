import random
from logging import getLogger
import os.path as osp
from typing import Any

import lightning.pytorch as pl
import optuna
from lightning.pytorch.utilities import rank_zero_only  # type: ignore
from optuna.integration import PyTorchLightningPruningCallback

from luxonis_train.callbacks import LuxonisProgressBar
from luxonis_train.models import LuxonisModel
from luxonis_train.utils import Config
from luxonis_train.utils.tracker import LuxonisTrackerPL

from .core import Core

logger = getLogger(__name__)


class Tuner(Core):
    def __init__(self, cfg: str | dict, args: list[str] | tuple[str, ...] | None):
        """Main API which is used to perform hyperparameter tunning.

        @type cfg: str | dict[str, Any] | Config
        @param cfg: Path to config file or config dict used to setup training.

        @type args: list[str] | tuple[str, ...] | None
        @param args: Argument dict provided through command line,
            used for config overriding.
        """
        super().__init__(cfg, args)
        if self.cfg.tuner is None:
            raise ValueError("You have to specify the `tuner` section in config.")
        self.tune_cfg = self.cfg.tuner

    def tune(self) -> None:
        """Runs Optuna tunning of hyperparameters."""

        pruner = (
            optuna.pruners.MedianPruner()
            if self.tune_cfg.use_pruner
            else optuna.pruners.NopPruner()
        )

        storage = None
        if self.tune_cfg.storage.active:
            if self.tune_cfg.storage.storage_type == "local":
                storage = "sqlite:///study_local.db"
            else:
                storage = "postgresql://{}:{}@{}:{}/{}".format(
                    self.cfg.ENVIRON.POSTGRES_USER,
                    self.cfg.ENVIRON.POSTGRES_PASSWORD,
                    self.cfg.ENVIRON.POSTGRES_HOST,
                    self.cfg.ENVIRON.POSTGRES_PORT,
                    self.cfg.ENVIRON.POSTGRES_DB,
                )

        study = optuna.create_study(
            study_name=self.tune_cfg.study_name,
            storage=storage,
            direction="minimize",
            pruner=pruner,
            load_if_exists=True,
        )

        study.optimize(
            self._objective,
            n_trials=self.tune_cfg.n_trials,
            timeout=self.tune_cfg.timeout,
        )

    def _objective(self, trial: optuna.trial.Trial) -> float:
        """Objective function used to optimize Optuna study."""
        rank = rank_zero_only.rank
        cfg_tracker = self.cfg.tracker
        tracker_params = cfg_tracker.model_dump()
        tracker = LuxonisTrackerPL(
            rank=rank,
            mlflow_tracking_uri=self.cfg.ENVIRON.MLFLOW_TRACKING_URI,
            is_sweep=True,
            **tracker_params,
        )
        run_save_dir = osp.join(cfg_tracker.save_directory, tracker.run_name)

        curr_params = self._get_trial_params(trial)
        curr_params["model.predefined_model"] = None

        cfg_copy = self.cfg.model_copy(deep=True)
        cfg_copy.trainer.preprocessing.augmentations = [
            a
            for a in cfg_copy.trainer.preprocessing.augmentations
            if a.name != "Normalize"
        ]  # manually remove Normalize so it doesn't duplicate it when creating new cfg instance
        Config.clear_instance()
        cfg = Config.get_config(cfg_copy.model_dump(), curr_params)

        tracker.log_hyperparams(curr_params)

        cfg.save_data(osp.join(run_save_dir, "config.yaml"))

        lightning_module = LuxonisModel(
            cfg=cfg,
            dataset_metadata=self.dataset_metadata,
            save_dir=run_save_dir,
            input_shape=self.loaders["train"].input_shape,
        )
        lightning_module._core = self
        pruner_callback = PyTorchLightningPruningCallback(
            trial, monitor="val_loss/loss"
        )
        callbacks: list[pl.Callback] = (
            [LuxonisProgressBar()] if self.cfg.use_rich_text else []
        )
        callbacks.append(pruner_callback)

        deterministic = False
        if self.cfg.trainer.seed:
            pl.seed_everything(cfg.trainer.seed, workers=True)
            deterministic = True

        pl_trainer = pl.Trainer(
            accelerator=cfg.trainer.accelerator,
            devices=cfg.trainer.devices,
            strategy=cfg.trainer.strategy,
            logger=tracker,  # type: ignore
            max_epochs=cfg.trainer.epochs,
            accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
            check_val_every_n_epoch=cfg.trainer.validation_interval,
            num_sanity_val_steps=cfg.trainer.num_sanity_val_steps,
            profiler=cfg.trainer.profiler,
            callbacks=callbacks,
            deterministic=deterministic,
        )

        pl_trainer.fit(
            lightning_module,  # type: ignore
            self.pytorch_loaders["train"],
            self.pytorch_loaders["val"],
        )
        pruner_callback.check_pruned()

        if "val/loss" not in pl_trainer.callback_metrics:
            raise ValueError(
                "No validation loss found. "
                "This can happen if `TestOnTrainEnd` callback is used."
            )

        return pl_trainer.callback_metrics["val/loss"].item()

    def _get_trial_params(self, trial: optuna.trial.Trial) -> dict[str, Any]:
        """Get trial params based on specified config."""
        cfg_tuner = self.tune_cfg.params
        new_params = {}
        for key, value in cfg_tuner.items():
            key_info = key.split("_")
            key_name = "_".join(key_info[:-1])
            key_type = key_info[-1]
            match key_type, value:
                case "subset", [list(whole_set), int(subset_size)]:
                    if key_name.split(".")[-1] != "augmentations":
                        raise ValueError(
                            "Subset sampling currently only supported for augmentations"
                        )
                    whole_set_indices = self._augs_to_indices(whole_set)
                    subset = random.sample(whole_set_indices, subset_size)
                    for aug_id in whole_set_indices:
                        new_params[f"{key_name}.{aug_id}.active"] = (
                            True if aug_id in subset else False
                        )
                    continue
                case "categorical", list(lst):
                    new_value = trial.suggest_categorical(key_name, lst)
                case "float", [float(low), float(high), *tail]:
                    step = tail[0] if tail else None
                    if step is not None and not isinstance(step, float):
                        raise ValueError(
                            f"Step for float type must be float, but got {step}"
                        )
                    new_value = trial.suggest_float(key_name, low, high, step=step)
                case "int", [int(low), int(high), *tail]:
                    step = tail[0] if tail else 1
                    if not isinstance(step, int):
                        raise ValueError(
                            f"Step for int type must be int, but got {step}"
                        )
                    new_value = trial.suggest_int(key_name, low, high, step=step)
                case "loguniform", [float(low), float(high)]:
                    new_value = trial.suggest_loguniform(key_name, low, high)
                case "uniform", [float(low), float(high)]:
                    new_value = trial.suggest_uniform(key_name, low, high)
                case _, _:
                    raise KeyError(
                        f"Combination of {key_type} and {value} not supported"
                    )

            new_params[key_name] = new_value

        if len(new_params) == 0:
            raise ValueError(
                "No paramteres to tune. Specify them under `tuner.params`."
            )
        return new_params

    def _augs_to_indices(self, aug_names: list[str]) -> list[int]:
        """Maps augmentation names to indices"""
        all_augs = self.cfg.trainer.preprocessing.augmentations
        aug_indices = []
        for aug_name in aug_names:
            if aug_name == "Normalize":
                logger.warn(
                    f"'{aug_name}' should should be tuned directly by adding '...normalize.active_categorical' to the tuner params, skipping."
                )
                continue
            index = [i for i, a in enumerate(all_augs) if a.name == aug_name]
            if len(index) == 0:
                logger.warn(
                    f"Augmentation '{aug_name}' not found under trainer augemntations, skipping."
                )
                continue
            aug_indices.append(index[0])
        return aug_indices
