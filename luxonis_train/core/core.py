import os
import os.path as osp
import signal
import threading
from contextlib import contextmanager, suppress
from logging import getLogger
from pathlib import Path
from typing import Any, Literal

import cv2
import lightning.pytorch as pl
import lightning_utilities.core.rank_zero as rank_zero_module
import rich.traceback
import torch
import torch.utils.data as torch_data
import yaml
from lightning.pytorch.utilities import rank_zero_only  # type: ignore  # type: ignore
from luxonis_ml.data import Augmentations
from luxonis_ml.utils import LuxonisFileSystem, reset_logging, setup_logging

from luxonis_train.attached_modules.visualizers import get_unnormalized_images
from luxonis_train.callbacks import LuxonisProgressBar
from luxonis_train.models import LuxonisLightningModule
from luxonis_train.utils.config import Config
from luxonis_train.utils.general import DatasetMetadata
from luxonis_train.utils.loaders import collate_fn
from luxonis_train.utils.registry import LOADERS
from luxonis_train.utils.tracker import LuxonisTrackerPL

from .utils.export_utils import (
    blobconverter_export,
    get_preprocessing,
    try_onnx_simplify,
)

logger = getLogger(__name__)


class LuxonisModel:
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

        if self.cfg.use_rich_text:
            rich.traceback.install(suppress=[pl, torch], show_locals=False)

        self.rank = rank_zero_only.rank

        self.tracker = self._create_tracker()
        # NOTE: tracker.experiment has to be called first in order
        # for the run_id to be initialized
        # TODO: it shouldn't be a property because of the above
        with suppress(Exception):
            _ = self.tracker.experiment
        self._run_id = self.tracker.run_id

        self.run_save_dir = os.path.join(
            self.cfg.tracker.save_directory, self.tracker.run_name
        )
        self.log_file = osp.join(self.run_save_dir, "luxonis_train.log")
        self.files_to_upload = [self.log_file]

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
        # if self.cfg.trainer.use_weighted_sampler:
        #     classes_count = self.loaders["train"].get_classes()[1]
        #     if len(classes_count) == 0:
        #         logger.warning(
        #             "WeightedRandomSampler only available for classification tasks. Using default sampler instead."
        #         )
        #     else:
        #         weights = [1 / i for i in classes_count.values()]
        #         num_samples = sum(classes_count.values())
        #         sampler = torch_data.WeightedRandomSampler(weights, num_samples)

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

        self.input_shape = self.loaders["train"].input_shape

        self.lightning_module = LuxonisLightningModule(
            cfg=self.cfg,
            dataset_metadata=self.dataset_metadata,
            save_dir=self.run_save_dir,
            input_shape=self.input_shape,
            _core=self,
        )

    def _upload_logs(self, tracker: LuxonisTrackerPL | None = None) -> None:
        tracker = tracker or self.tracker
        if self.cfg.tracker.is_mlflow:
            logger.info("Uploading logs to MLFlow.")
            fs = LuxonisFileSystem(
                "mlflow://",
                allow_active_mlflow_run=True,
                allow_local=False,
            )
            fs.put_file(
                local_path=self.log_file,
                remote_path="luxonis_train.log",
                mlflow_instance=tracker.experiment.get("mlflow"),
            )

    def _train(self, resume: str | None, *args, **kwargs):
        try:
            self.pl_trainer.fit(*args, ckpt_path=resume, **kwargs)
        except Exception:
            logger.exception("Encountered exception during training.")
        finally:
            self._upload_logs(self._create_tracker(self._run_id))

    def train(self, new_thread: bool = False, resume: str | None = None) -> None:
        """Runs training.

        @type new_thread: bool
        @param new_thread: Runs training in new thread if set to True.
        """

        if self.cfg.trainer.matmul_precision is not None:
            logger.info(
                f"Setting matmul precision to {self.cfg.trainer.matmul_precision}"
            )
            torch.set_float32_matmul_precision(self.cfg.trainer.matmul_precision)

        if resume is not None:
            resume = str(LuxonisFileSystem.download(resume, self.run_save_dir))

        def graceful_exit(signum: int, _):
            logger.info(f"{signal.Signals(signum).name} received, stopping training...")
            ckpt_path = osp.join(self.run_save_dir, "resume.ckpt")
            self.pl_trainer.save_checkpoint(ckpt_path)
            tracker = self._create_tracker(self._run_id)
            self._upload_logs(tracker)

            if self.cfg.tracker.is_mlflow:
                logger.info("Uploading checkpoint to MLFlow.")
                fs = LuxonisFileSystem(
                    "mlflow://",
                    allow_active_mlflow_run=True,
                    allow_local=False,
                )
                fs.put_file(
                    local_path=ckpt_path,
                    remote_path="resume.ckpt",
                    mlflow_instance=tracker.experiment.get("mlflow"),
                )

            exit(0)

        signal.signal(signal.SIGTERM, graceful_exit)

        if not new_thread:
            logger.info(f"Checkpoints will be saved in: {self.run_save_dir}")
            logger.info("Starting training...")
            self._train(
                resume,
                self.lightning_module,
                self.pytorch_loaders["train"],
                self.pytorch_loaders["val"],
            )
            logger.info("Training finished")
            logger.info(f"Checkpoints saved in: {self.run_save_dir}")

        else:
            # Every time exception happens in the Thread, this hook will activate
            def thread_exception_hook(args):
                self.error_message = str(args.exc_value)

            threading.excepthook = thread_exception_hook

            self.thread = threading.Thread(
                target=self._train,
                args=(
                    resume,
                    self.lightning_module,
                    self.pytorch_loaders["train"],
                    self.pytorch_loaders["val"],
                ),
                daemon=True,
            )
            self.thread.start()

    @contextmanager
    def use_weights(self, weights: str | None = None):
        old_weights = None
        if weights is not None:
            old_weights = self.lightning_module.state_dict()
            self.lightning_module.load_checkpoint(weights)

        yield

        if old_weights is not None:
            try:
                self.lightning_module.load_state_dict(old_weights)
            except RuntimeError:
                logger.error(
                    "Failed to strictly load old weights. The model likey underwent reparametrization, "
                    "which is a destructive operation. Loading old weights with strict=False."
                )
                self.lightning_module.load_state_dict(old_weights, strict=False)
            del old_weights

    def export(
        self, onnx_save_path: str | None = None, weights: str | None = None
    ) -> None:
        """Runs export.

        @type onnx_path: str | None
        @param onnx_path: Path to .onnx model. If not specified, model will be saved
            to export directory with name specified in config file.

        @raises RuntimeError: If `onnxsim` fails to simplify the model.
        """

        weights = weights or self.cfg.model.weights

        if weights is None:
            logger.warning(
                "No model weights specified. Exporting model without weights."
            )

        export_save_dir = Path(self.run_save_dir, "export")
        export_path = export_save_dir / (
            self.cfg.exporter.model_name or self.cfg.model.name
        )

        if not export_save_dir.exists():
            export_save_dir.mkdir(parents=True, exist_ok=True)

        onnx_save_path = onnx_save_path or str(export_path.with_suffix(".onnx"))

        with self.use_weights(weights):
            output_names = self.lightning_module.export_onnx(
                onnx_save_path, **self.cfg.exporter.onnx.model_dump()
            )

        try_onnx_simplify(onnx_save_path)

        files_to_upload = [weights, onnx_save_path]

        scale_values, mean_values, reverse_channels = get_preprocessing(self.cfg)

        if self.cfg.exporter.blobconverter.active:
            try:
                blobconverter_export(
                    self.cfg.exporter,
                    scale_values,
                    mean_values,
                    reverse_channels,
                    str(export_save_dir),
                    onnx_save_path,
                )
            except ImportError:
                logger.error("Failed to import `blobconverter`")
                logger.warning(
                    "`blobconverter` not installed. Skipping .blob model conversion. "
                    "Ensure `blobconverter` is installed in your environment."
                )

        modelconverter_config = {
            "input_model": onnx_save_path,
            "scale_values": scale_values,
            "mean_values": mean_values,
            "reverse_input_channels": reverse_channels,
            "shape": list(self.input_shape),
            "data_type": self.cfg.exporter.data_type,
            "outputs": [{"name": name} for name in output_names],
        }

        with open(export_path.with_suffix(".yaml"), "w") as f:
            yaml.dump(modelconverter_config, f)
            files_to_upload.append(f.name)

        if self.cfg.exporter.upload_url is not None:
            self.files_to_upload.extend(files_to_upload)

    def test(
        self, new_thread: bool = False, view: Literal["train", "val", "test"] = "test"
    ) -> None:
        """Runs testing.

        @type new_thread: bool
        @param new_thread: Runs testing in new thread if set to True.
        """

        if view not in self.pytorch_loaders:
            raise ValueError(
                f"View {view} is not valid. Valid views are: 'train', 'val', 'test'."
            )
        loader = self.pytorch_loaders[view]

        if not new_thread:
            self.pl_trainer.test(self.lightning_module, loader)
        else:
            self.thread = threading.Thread(
                target=self.pl_trainer.test,
                args=(self.lightning_module, loader),
                daemon=True,
            )
            self.thread.start()

    def infer(
        self,
        view: Literal["train", "val", "test"] = "val",
        save_dir: str | Path | None = None,
    ) -> None:
        self.lightning_module.eval()
        save_dir = Path(save_dir) if save_dir is not None else None
        if save_dir is not None:
            save_dir.mkdir(exist_ok=True, parents=True)

        k = 0
        if view not in self.pytorch_loaders:
            raise ValueError(
                f"View {view} is not valid. Valid views are: 'train', 'val', 'test'."
            )
        for inputs, labels in self.pytorch_loaders[view]:
            images = get_unnormalized_images(self.cfg, inputs)
            outputs = self.lightning_module.forward(
                inputs, labels, images=images, compute_visualizations=True
            )

            for node_name, visualizations in outputs.visualizations.items():
                for viz_name, viz_batch in visualizations.items():
                    for i, viz in enumerate(viz_batch):
                        viz_arr = viz.detach().cpu().numpy().transpose(1, 2, 0)
                        viz_arr = cv2.cvtColor(viz_arr, cv2.COLOR_RGB2BGR)
                        name = f"{node_name}/{viz_name}/{i}"
                        if save_dir is not None:
                            name = name.replace("/", "_")
                            cv2.imwrite(str(save_dir / f"{name}_{k}.png"), viz_arr)
                            k += 1
                        else:
                            cv2.imshow(name, viz_arr)

            if save_dir is None:
                if cv2.waitKey(0) == ord("q"):
                    exit()

    def tune(self) -> None:
        """Runs Optuna tunning of hyperparameters."""
        import optuna
        from optuna.integration import PyTorchLightningPruningCallback

        from .utils.tuner_utils import get_trial_params

        def _objective(trial: optuna.trial.Trial) -> float:
            """Objective function used to optimize Optuna study."""
            rank = rank_zero_only.rank
            cfg_tracker = self.cfg.tracker
            tracker_params = cfg_tracker.model_dump()
            child_tracker = LuxonisTrackerPL(
                rank=rank,
                mlflow_tracking_uri=self.cfg.ENVIRON.MLFLOW_TRACKING_URI,
                is_sweep=True,
                **tracker_params,
            )

            run_save_dir = osp.join(cfg_tracker.save_directory, child_tracker.run_name)

            assert self.cfg.tuner is not None
            curr_params = get_trial_params(all_augs, self.cfg.tuner.params, trial)
            curr_params["model.predefined_model"] = None

            cfg_copy = self.cfg.model_copy(deep=True)
            cfg_copy.trainer.preprocessing.augmentations = [
                a
                for a in cfg_copy.trainer.preprocessing.augmentations
                if a.name != "Normalize"
            ]  # manually remove Normalize so it doesn't duplicate it when creating new cfg instance
            Config.clear_instance()
            cfg = Config.get_config(cfg_copy.model_dump(), curr_params)

            child_tracker.log_hyperparams(curr_params)

            cfg.save_data(osp.join(run_save_dir, "config.yaml"))

            lightning_module = LuxonisLightningModule(
                cfg=cfg,
                dataset_metadata=self.dataset_metadata,
                save_dir=run_save_dir,
                input_shape=self.loaders["train"].input_shape,
                _core=self,
            )
            callbacks: list[pl.Callback] = (
                [LuxonisProgressBar()] if self.cfg.use_rich_text else []
            )
            pruner_callback = PyTorchLightningPruningCallback(trial, monitor="val/loss")
            callbacks.append(pruner_callback)
            deterministic = False
            if self.cfg.trainer.seed:
                pl.seed_everything(cfg.trainer.seed, workers=True)
                deterministic = True

            pl_trainer = pl.Trainer(
                accelerator=cfg.trainer.accelerator,
                devices=cfg.trainer.devices,
                strategy=cfg.trainer.strategy,
                logger=child_tracker,  # type: ignore
                max_epochs=cfg.trainer.epochs,
                accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
                check_val_every_n_epoch=cfg.trainer.validation_interval,
                num_sanity_val_steps=cfg.trainer.num_sanity_val_steps,
                profiler=cfg.trainer.profiler,
                callbacks=callbacks,
                deterministic=deterministic,
            )

            try:
                pl_trainer.fit(
                    lightning_module,  # type: ignore
                    self.pytorch_loaders["val"],
                    self.pytorch_loaders["train"],
                )

                pruner_callback.check_pruned()

            except optuna.TrialPruned as e:
                # Pruning is done by raising an error
                logger.info(e)

            if "val/loss" not in pl_trainer.callback_metrics:
                raise ValueError(
                    "No validation loss found. "
                    "This can happen if `TestOnTrainEnd` callback is used."
                )

            return pl_trainer.callback_metrics["val/loss"].item()

        cfg_tuner = self.cfg.tuner
        if cfg_tuner is None:
            raise ValueError("You have to specify the `tuner` section in config.")

        all_augs = [a.name for a in self.cfg.trainer.preprocessing.augmentations]
        rank = rank_zero_only.rank
        cfg_tracker = self.cfg.tracker
        tracker_params = cfg_tracker.model_dump()
        # NOTE: wandb doesn't allow multiple concurrent runs, handle this separately
        tracker_params["is_wandb"] = False
        self.parent_tracker = LuxonisTrackerPL(
            rank=rank,
            mlflow_tracking_uri=self.cfg.ENVIRON.MLFLOW_TRACKING_URI,
            is_sweep=False,
            **tracker_params,
        )
        if self.parent_tracker.is_mlflow:
            # Experiment needs to be interacted with to create actual MLFlow run
            self.parent_tracker.experiment["mlflow"].active_run()

        logger.info("Starting tuning...")

        pruner = (
            optuna.pruners.MedianPruner()
            if cfg_tuner.use_pruner
            else optuna.pruners.NopPruner()
        )

        storage = None
        if cfg_tuner.storage.active:
            if cfg_tuner.storage.storage_type == "local":
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
            study_name=cfg_tuner.study_name,
            storage=storage,
            direction="minimize",
            pruner=pruner,
            load_if_exists=cfg_tuner.continue_existing_study,
        )

        study.optimize(
            _objective,
            n_trials=cfg_tuner.n_trials,
            timeout=cfg_tuner.timeout,
        )

        best_study_params = study.best_params
        logger.info(f"Best study parameters: {best_study_params}")

        self.parent_tracker.log_hyperparams(best_study_params)

        if self.cfg.tracker.is_wandb:
            # If wandb used then init parent tracker separately at the end
            wandb_parent_tracker = LuxonisTrackerPL(
                project_name=self.cfg.tracker.project_name,
                project_id=self.cfg.tracker.project_id,
                run_name=self.parent_tracker.run_name,
                save_directory=self.cfg.tracker.save_directory,
                is_wandb=True,
                wandb_entity=self.cfg.tracker.wandb_entity,
                rank=rank_zero_only.rank,
            )
            wandb_parent_tracker.log_hyperparams(best_study_params)

    def set_train_augmentations(self, aug: Augmentations) -> None:
        """Sets augmentations used for training dataset."""
        self.train_augmentations = aug

    def set_val_augmentations(self, aug: Augmentations) -> None:
        """Sets augmentations used for validation dataset."""
        self.val_augmentations = aug

    def set_test_augmentations(self, aug: Augmentations) -> None:
        """Sets augmentations used for test dataset."""
        self.test_augmentations = aug

    def _create_tracker(self, run_id: str | None = None) -> LuxonisTrackerPL:
        kwargs = self.cfg.tracker.model_dump()
        if run_id is not None:
            kwargs["run_id"] = run_id
        return LuxonisTrackerPL(
            rank=self.rank,
            mlflow_tracking_uri=self.cfg.ENVIRON.MLFLOW_TRACKING_URI,
            **kwargs,
        )

    @rank_zero_only
    def get_status(self) -> tuple[int, int]:
        """Get current status of training.

        @rtype: tuple[int, int]
        @return: First element is current epoch, second element is total number of
            epochs.
        """
        return self.lightning_module.get_status()

    @rank_zero_only
    def get_status_percentage(self) -> float:
        """Return percentage of current training, takes into account early stopping.

        @rtype: float
        @return: Percentage of current training in range 0-100.
        """
        return self.lightning_module.get_status_percentage()

    @rank_zero_only
    def get_error_message(self) -> str | None:
        """Return error message if one occurs while running in thread, otherwise None.

        @rtype: str | None
        @return: Error message
        """
        return self.error_message

    @rank_zero_only
    def get_min_loss_checkpoint_path(self) -> str | None:
        """Return best checkpoint path with respect to minimal validation loss.

        @rtype: str
        @return: Path to best checkpoint with respect to minimal validation loss
        """
        return self.pl_trainer.checkpoint_callbacks[0].best_model_path  # type: ignore

    @rank_zero_only
    def get_best_metric_checkpoint_path(self) -> str | None:
        """Return best checkpoint path with respect to best validation metric.

        @rtype: str
        @return: Path to best checkpoint with respect to best validation metric
        """
        return self.pl_trainer.checkpoint_callbacks[1].best_model_path  # type: ignore
