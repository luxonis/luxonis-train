import signal
import sys
import threading
from collections.abc import Mapping
from pathlib import Path
from threading import ExceptHookArgs
from typing import Any, Literal, TypeAlias, overload

import lightning.pytorch as pl
import lightning_utilities.core.rank_zero as rank_zero_module
import rich.traceback
import torch
import torch.utils.data as torch_data
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.utilities import rank_zero_only
from loguru import logger
from luxonis_ml.nn_archive import ArchiveGenerator
from luxonis_ml.nn_archive.config import CONFIG_VERSION
from luxonis_ml.typing import Params, PathType
from luxonis_ml.utils import LuxonisFileSystem
from typeguard import typechecked

from luxonis_train.callbacks import (
    LuxonisRichProgressBar,
    LuxonisTQDMProgressBar,
)
from luxonis_train.config import Config
from luxonis_train.lightning import LuxonisLightningModule
from luxonis_train.loaders import BaseLoaderTorch, collate_fn
from luxonis_train.loaders.luxonis_loader_torch import LuxonisLoaderTorch
from luxonis_train.registry import LOADERS
from luxonis_train.utils import (
    DatasetMetadata,
    LuxonisTrackerPL,
    setup_logging,
)

from .utils.export_utils import (
    blobconverter_export,
    get_preprocessing,
    replace_weights,
    try_onnx_simplify,
)
from .utils.infer_utils import (
    IMAGE_FORMATS,
    VIDEO_FORMATS,
    infer_from_dataset,
    infer_from_directory,
    infer_from_video,
)
from .utils.train_utils import create_trainer

View: TypeAlias = Literal["train", "val", "test"]


class LuxonisModel:
    """Common logic of the core components.

    This class contains common logic of the core components (trainer,
    evaluator, exporter, etc.).
    """

    def __init__(
        self,
        cfg: str | Params | Config | None,
        opts: Params | list[str] | tuple[str, ...] | None = None,
    ):
        """Constructs a new Core instance.

        Loads the config and initializes loaders, dataloaders, augmentations,
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

        self.cfg_preprocessing = self.cfg.trainer.preprocessing

        rich.traceback.install(suppress=[pl, torch], show_locals=False)

        self.tracker = LuxonisTrackerPL(
            rank=rank_zero_only.rank,
            mlflow_tracking_uri=self.cfg.ENVIRON.MLFLOW_TRACKING_URI,
            _auto_finalize=False,
            **self.cfg.tracker.model_dump(),
        )

        self.run_save_dir = (
            self.cfg.tracker.save_directory / self.tracker.run_name
        )
        self.log_file = self.run_save_dir / "luxonis_train.log"
        self.error_message = None

        setup_logging(file=self.log_file)

        # NOTE: overriding logger in pl so it uses our logger to log device info
        rank_zero_module.log = logger

        if self.cfg.trainer.seed is not None:
            pl.seed_everything(self.cfg.trainer.seed, workers=True)

        self.pl_trainer = create_trainer(
            self.cfg.trainer,
            logger=self.tracker,
            callbacks=(
                LuxonisRichProgressBar()
                if self.cfg.trainer.use_rich_progress_bar
                else LuxonisTQDMProgressBar()
            ),
            precision=self.cfg.trainer.precision,
        )

        self.loaders: dict[View, BaseLoaderTorch] = {}
        loader_name = self.cfg.loader.name
        Loader = LOADERS.get(loader_name)
        for view in ("train", "val", "test"):
            if (
                view != "train"
                and issubclass(Loader, LuxonisLoaderTorch)
                and self.cfg.loader.params.get("dataset_dir") is not None
            ):
                self.cfg.loader.params["delete_existing"] = False

            self.loaders[view] = Loader(
                view={
                    "train": self.cfg.loader.train_view,
                    "val": self.cfg.loader.val_view,
                    "test": self.cfg.loader.test_view,
                }[view],
                image_source=self.cfg.loader.image_source,
                height=self.cfg_preprocessing.train_image_size.height,
                width=self.cfg_preprocessing.train_image_size.width,
                augmentation_config=self.cfg_preprocessing.get_active_augmentations(),
                color_space=self.cfg_preprocessing.color_space,
                keep_aspect_ratio=self.cfg_preprocessing.keep_aspect_ratio,
                **self.cfg.loader.params,  # type: ignore
            )

        for name, loader in self.loaders.items():
            logger.info(
                f"{name.capitalize()} loader - view: {loader.view}, size: {len(loader)}"
            )
            if len(loader) == 0:
                logger.warning(f"{name.capitalize()} loader is empty!")

        sampler = None
        # TODO: implement weighted sampler
        if self.cfg.trainer.use_weighted_sampler:
            raise NotImplementedError(
                "Weighted sampler is not implemented yet."
            )

        self.pytorch_loaders: dict[View, torch_data.DataLoader] = {}
        for view in ("train", "val", "test"):
            if self.cfg.trainer.n_validation_batches is not None and view in {
                "val",
                "test",
            }:
                generator = torch.Generator()
                generator.manual_seed(self.cfg.trainer.seed or 42)
                if self.cfg.trainer.n_validation_batches == -1:
                    self.cfg.trainer.n_validation_batches = len(
                        self.loaders["val"]
                    )
                    n_samples = len(self.loaders[view])
                else:
                    n_samples = (
                        self.cfg.trainer.n_validation_batches
                        * self.cfg.trainer.batch_size
                    )
                indices = range(min(n_samples, len(self.loaders[view])))
                loader = torch_data.Subset(self.loaders[view], indices)
            else:
                loader = self.loaders[view]

            self.pytorch_loaders[view] = torch_data.DataLoader(
                loader,
                batch_size=self.cfg.trainer.batch_size,
                num_workers=self.cfg.trainer.n_workers,
                collate_fn=collate_fn,
                shuffle=view == "train",
                drop_last=(
                    self.cfg.trainer.skip_last_batch
                    if view == "train"
                    else False
                ),
                pin_memory=self.cfg.trainer.pin_memory,
                sampler=sampler if view == "train" else None,
                generator=generator
                if (
                    self.cfg.trainer.n_validation_batches is not None
                    and view in ["val", "test"]
                )
                else None,
            )

        self.dataset_metadata = DatasetMetadata.from_loader(
            self.loaders["train"]
        )
        self.config_file = self.run_save_dir / "training_config.yaml"
        self.cfg.save_data(self.config_file)

        self.input_shapes = self.loaders["train"].input_shapes

        self.lightning_module = LuxonisLightningModule(
            cfg=self.cfg,
            dataset_metadata=self.dataset_metadata,
            save_dir=self.run_save_dir,
            input_shapes=self.input_shapes,
            _core=self,
        )

        self._exported_models: dict[str, Path] = {}

    def _train(self, resume: PathType | None, *args, **kwargs) -> None:
        status = "success"
        try:
            self.pl_trainer.fit(*args, ckpt_path=resume, **kwargs)
        except Exception:  # pragma: no cover
            logger.exception("Encountered an exception during training.")
            status = "failed"
            raise
        finally:
            self.tracker.upload_artifact(self.log_file, typ="logs")
            self.tracker.upload_artifact(self.config_file, typ="config")
            self.tracker._finalize(status)

    def train(
        self, new_thread: bool = False, weights: PathType | None = None
    ) -> None:
        """Runs training.

        @type new_thread: bool
        @param new_thread: Runs training in new thread if set to True.
        @type weights: str | None
        @param weights: Path to the weights. If user specifies weights
            in the config file, the weights provided here will take
            precedence.
        """

        if self.cfg.trainer.matmul_precision is not None:
            logger.info(
                f"Setting matmul precision to {self.cfg.trainer.matmul_precision}"
            )
            torch.set_float32_matmul_precision(
                self.cfg.trainer.matmul_precision
            )

        if weights is not None:
            weights = LuxonisFileSystem.download(
                str(weights), self.run_save_dir
            )
            if self.cfg.model.weights is not None:
                logger.warning(
                    "Weights provided in the command line, but config weights are set. "
                    "Ignoring weights provided in config."
                )
            self.lightning_module.load_checkpoint(weights)
        else:
            weights = self.cfg.model.weights

        resume_weights = weights if self.cfg.trainer.resume_training else None

        if self.cfg.trainer.resume_training and resume_weights is None:
            logger.warning(
                "Resume training is enabled but no weights were provided. "
                "Training will start from scratch."
            )

        def graceful_exit(signum: int, _: Any) -> None:  # pragma: no cover
            logger.info(
                f"{signal.Signals(signum).name} received, stopping training..."
            )
            ckpt_path = self.run_save_dir / "resume.ckpt"
            self.pl_trainer.save_checkpoint(ckpt_path)
            self.tracker.upload_artifact(
                ckpt_path, typ="checkpoints", name="resume.ckpt"
            )
            self.tracker._finalize(status="failed")
            sys.exit()

        signal.signal(signal.SIGTERM, graceful_exit)

        if not new_thread:
            logger.info(f"Checkpoints will be saved in: {self.run_save_dir}")
            logger.info("Starting training...")
            self._train(
                resume_weights,
                self.lightning_module,
                self.pytorch_loaders["train"],
                self.pytorch_loaders["val"],
            )
            logger.info("Training finished")
            logger.info(f"Checkpoints saved in: {self.run_save_dir}")

        else:  # pragma: no cover
            # Every time exception happens in the Thread, this hook will activate
            def thread_exception_hook(args: ExceptHookArgs) -> None:
                self.error_message = str(args.exc_value)

            threading.excepthook = thread_exception_hook

            self.thread = threading.Thread(
                target=self._train,
                args=(
                    resume_weights,
                    self.lightning_module,
                    self.pytorch_loaders["train"],
                    self.pytorch_loaders["val"],
                ),
                daemon=True,
            )
            self.thread.start()

    def export(
        self,
        onnx_save_path: PathType | None = None,
        weights: PathType | None = None,
        ignore_missing_weights: bool = False,
    ) -> None:
        """Runs export.

        @type onnx_save_path: PathType | None
        @param onnx_save_path: Path to where the exported ONNX model will be saved.
            If not specified, model will be saved to the export directory
            with the name specified in config file.
        @type weights: PathType | None
        @param weights: Path to the checkpoint from which to load weights.
            If not specified, the value of `model.weights` from the
            configuration file will be used. The current weights of the
            model will be temporarily replaced with the weights from the
            specified checkpoint.
        @type ignore_missing_weights: bool
        @param ignore_missing_weights: If set to True, the warning about
            missing weights will be suppressed.
        @raises RuntimeError: If C{onnxsim} fails to simplify the model.
        """

        weights = weights or self.cfg.model.weights

        if not ignore_missing_weights and weights is None:
            logger.warning(
                "No model weights specified. Exporting model without weights."
            )

        export_save_dir = Path(self.run_save_dir, "export")
        export_save_dir.mkdir(parents=True, exist_ok=True)

        export_path = export_save_dir / (
            self.cfg.exporter.name or self.cfg.model.name
        )
        onnx_save_path = str(
            onnx_save_path or export_path.with_suffix(".onnx")
        )

        with replace_weights(self.lightning_module, weights):
            output_names = self.lightning_module.export_onnx(
                onnx_save_path, **self.cfg.exporter.onnx.model_dump()
            )

        try_onnx_simplify(onnx_save_path)
        self._exported_models["onnx"] = Path(onnx_save_path)

        scale, mean, color_space = get_preprocessing(
            self.cfg_preprocessing, "Model export"
        )
        scale_values = self.cfg.exporter.scale_values or scale
        mean_values = self.cfg.exporter.mean_values or mean
        if self.cfg.exporter.reverse_input_channels is not None:
            reverse_input_channels = self.cfg.exporter.reverse_input_channels
        else:
            logger.info(
                "`exporter.reverse_input_channels` not specified. "
                "Using the `trainer.preprocessing.color_space` value "
                "to determine if the channels should be reversed. "
                f"`color_space` = '{color_space}' -> "
                f"`reverse_input_channels` = `{color_space == 'RGB'}`"
            )
            reverse_input_channels = color_space == "RGB"

        if self.cfg.exporter.blobconverter.active:
            try:
                self._exported_models["blob"] = blobconverter_export(
                    self.cfg.exporter,
                    scale_values,
                    mean_values,
                    reverse_input_channels,
                    str(export_save_dir),
                    onnx_save_path,
                )
            except ImportError:
                logger.error("Failed to import `blobconverter`")
                logger.warning(
                    "`blobconverter` not installed. Skipping .blob model conversion. "
                    "Ensure `blobconverter` is installed in your environment."
                )

        if len(self.input_shapes) > 1:
            logger.error(
                "Generating modelconverter config for a model "
                "with multiple inputs is not implemented yet."
            )
            return

        modelconverter_config = {
            "input_model": onnx_save_path,
            "scale_values": scale_values,
            "mean_values": mean_values,
            "reverse_input_channels": reverse_input_channels,
            "shape": [1, *next(iter(self.input_shapes.values()))],
            "outputs": [{"name": name} for name in output_names],
        }

        for path in self._exported_models.values():
            if self.cfg.exporter.upload_to_run:
                self.tracker.upload_artifact(path, typ="export")
            if self.cfg.exporter.upload_url is not None:  # pragma: no cover
                LuxonisFileSystem.upload(path, self.cfg.exporter.upload_url)

        with open(export_path.with_suffix(".yaml"), "w") as f:
            yaml.safe_dump(modelconverter_config, f)
            if self.cfg.exporter.upload_to_run:
                self.tracker.upload_artifact(f.name, name=f.name, typ="export")
            if self.cfg.exporter.upload_url is not None:  # pragma: no cover
                LuxonisFileSystem.upload(f.name, self.cfg.exporter.upload_url)

    @overload
    def test(
        self,
        new_thread: Literal[False] = ...,
        view: Literal["train", "test", "val"] = "val",
        weights: PathType | None = ...,
    ) -> Mapping[str, float]: ...

    @overload
    def test(
        self,
        new_thread: Literal[True] = ...,
        view: Literal["train", "test", "val"] = "test",
        weights: PathType | None = ...,
    ) -> None: ...

    @typechecked
    def test(
        self,
        new_thread: bool = False,
        view: Literal["train", "val", "test"] = "test",
        weights: PathType | None = None,
    ) -> Mapping[str, float] | None:
        """Runs testing.

        @type new_thread: bool
        @param new_thread: Runs testing in a new thread if set to True.
        @type view: Literal["train", "test", "val"]
        @param view: Which view to run the testing on. Defauls to "test".
        @rtype: Mapping[str, float] | None
        @return: If new_thread is False, returns a dictionary test
            results.
        @type weights: PathType | None
        @param weights: Path to the checkpoint from which to load weights.
            If not specified, the value of `model.weights` from the
            configuration file will be used. The current weights of the
            model will be temporarily replaced with the weights from the
            specified checkpoint.
        """

        weights = weights or self.cfg.model.weights
        loader = self.pytorch_loaders[view]

        with replace_weights(self.lightning_module, weights):
            if new_thread:  # pragma: no cover
                self.thread = threading.Thread(
                    target=self.pl_trainer.test,
                    args=(self.lightning_module, loader),
                    daemon=True,
                )
                return self.thread.start()
            return self.pl_trainer.test(self.lightning_module, loader)[0]

    def infer(
        self,
        view: Literal["train", "val", "test"] = "val",
        save_dir: PathType | None = None,
        source_path: PathType | None = None,
        weights: PathType | None = None,
    ) -> None:
        """Runs inference.

        @type view: str
        @param view: Which split to run the inference on. Valid values
            are: C{"train"}, C{"val"}, C{"test"}. Defaults to C{"val"}.
        @type save_dir: PathType | None
        @param save_dir: Directory where to save the visualizations. If
            not specified, visualizations will be rendered on the
            screen.
        @type source_path: PathType | None
        @param source_path: Path to the image file, video file or directory.
            If None, defaults to using dataset images.
        @type weights: PathType | None
        @param weights: Path to the checkpoint from which to load weights.
            If not specified, the value of `model.weights` from the
            configuration file will be used. The current weights of the
            model will be temporarily replaced with the weights from the
            specified checkpoint.
        """
        self.lightning_module.eval()
        weights = weights or self.cfg.model.weights

        with replace_weights(self.lightning_module, weights):
            if save_dir is not None:
                save_dir = Path(save_dir)
                save_dir.mkdir(parents=True, exist_ok=True)
            if source_path is not None:
                source_path = Path(source_path)
                if source_path.suffix.lower() in VIDEO_FORMATS:
                    infer_from_video(
                        self, video_path=source_path, save_dir=save_dir
                    )
                elif source_path.is_file():
                    infer_from_directory(self, [source_path], save_dir)
                elif source_path.is_dir():
                    image_files = (
                        f
                        for f in source_path.iterdir()
                        if f.suffix.lower() in IMAGE_FORMATS
                    )
                    infer_from_directory(self, image_files, save_dir)
                else:
                    raise ValueError(
                        f"Source path {source_path} is not a valid file or directory."
                    )
            else:
                infer_from_dataset(self, view, save_dir)

    def tune(self) -> None:
        """Runs Optuna tuning of hyperparameters."""
        import optuna
        from optuna.integration import PyTorchLightningPruningCallback

        from .utils.tune_utils import get_trial_params

        def _objective(trial: optuna.trial.Trial) -> float:
            """Objective function used to optimize Optuna study."""
            cfg_tracker = self.cfg.tracker
            tracker_params = cfg_tracker.model_dump()
            child_tracker = LuxonisTrackerPL(
                rank=rank_zero_only.rank,
                mlflow_tracking_uri=self.cfg.ENVIRON.MLFLOW_TRACKING_URI,
                is_sweep=True,
                **tracker_params,
            )

            run_save_dir = cfg_tracker.save_directory / child_tracker.run_name

            assert self.cfg.tuner is not None
            curr_params = get_trial_params(
                all_augs, self.cfg.tuner.params, trial
            )
            curr_params["model.predefined_model"] = None

            cfg_copy = self.cfg.model_copy(deep=True)
            # manually remove Normalize so it doesn't
            # get duplicated when creating new cfg instance
            cfg_copy.trainer.preprocessing.augmentations = [
                a
                for a in cfg_copy.trainer.preprocessing.augmentations
                if a.name != "Normalize"
            ]
            cfg = Config.get_config(cfg_copy.model_dump(), curr_params)

            unsupported_callbacks = {
                "UploadCheckpoint",
                "ExportOnTrainEnd",
                "ArchiveOnTrainEnd",
                "TestOnTrainEnd",
            }

            filtered_callbacks = []
            for cb in cfg.trainer.callbacks:
                if cb.name in unsupported_callbacks:
                    logger.warning(
                        f"Callback '{cb.name}' is not supported for tunning and is removed from the callbacks list."
                    )
                else:
                    filtered_callbacks.append(cb)

            cfg.trainer.callbacks = filtered_callbacks

            child_tracker.log_hyperparams(curr_params)

            cfg.save_data(run_save_dir / "training_config.yaml")

            lightning_module = LuxonisLightningModule(
                cfg=cfg,
                dataset_metadata=self.dataset_metadata,
                save_dir=run_save_dir,
                input_shapes=self.loaders["train"].input_shapes,
                _core=self,
            )
            callbacks = [
                (
                    LuxonisRichProgressBar()
                    if cfg.trainer.use_rich_progress_bar
                    else LuxonisTQDMProgressBar()
                )
            ]

            pruner_callback = PyTorchLightningPruningCallback(
                trial, monitor="val/loss"
            )
            callbacks.append(pruner_callback)

            if self.cfg.trainer.seed is not None:
                pl.seed_everything(cfg.trainer.seed, workers=True)

            pl_trainer = create_trainer(
                cfg.trainer, logger=child_tracker, callbacks=callbacks
            )

            try:
                pl_trainer.fit(
                    lightning_module,
                    self.pytorch_loaders["train"],
                    self.pytorch_loaders["val"],
                )
                pruner_callback.check_pruned()

            # Pruning is done by raising an error
            except optuna.TrialPruned as e:
                logger.info(e)

            return pl_trainer.callback_metrics["val/loss"].item()

        cfg_tuner = self.cfg.tuner
        if cfg_tuner is None:
            raise ValueError(
                "You have to specify the `tuner` section in config."
            )

        all_augs = [a.name for a in self.cfg_preprocessing.augmentations]
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
        if self.parent_tracker.is_mlflow:  # pragma: no cover
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
            else:  # pragma: no cover
                storage = "postgresql://{}:{}@{}:{}/{}".format(  # noqa: UP032
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
            _objective, n_trials=cfg_tuner.n_trials, timeout=cfg_tuner.timeout
        )

        logger.info(f"Best study parameters: {study.best_params}")

        self.parent_tracker.log_hyperparams(study.best_params)

        if self.cfg.tracker.is_wandb:  # pragma: no cover
            # If wandb used then init parent tracker separately at the end
            wandb_parent_tracker = LuxonisTrackerPL(
                rank=rank_zero_only.rank,
                **(
                    self.cfg.tracker.model_dump()
                    | {"run_name": self.parent_tracker.run_name}
                ),
            )
            wandb_parent_tracker.log_hyperparams(study.best_params)

    def archive(
        self, path: PathType | None = None, weights: PathType | None = None
    ) -> Path:
        """Generates an NN Archive out of a model executable.

        @type path: PathType | None
        @param path: Path to the model executable. If not specified, the
            model will be exported first.
        @type weights: PathType | None
        @param weights: Path to the checkpoint from which to load weights.
            If not specified, the value of `model.weights` from the
            configuration file will be used. The current weights of the
            model will be temporarily replaced with the weights from the
            specified checkpoint.
        @rtype: Path
        @return: Path to the generated NN Archive.
        """
        weights = weights or self.cfg.model.weights
        with replace_weights(self.lightning_module, weights):
            return self._archive(path)

    def _archive(self, path: PathType | None = None) -> Path:
        from .utils.archive_utils import (
            get_head_configs,
            get_inputs,
            get_outputs,
        )

        archive_name = self.cfg.archiver.name or self.cfg.model.name
        archive_save_directory = Path(self.run_save_dir, "archive")
        archive_save_directory.mkdir(parents=True, exist_ok=True)
        inputs = []
        outputs = []

        if path is None:
            logger.warning("No model executable specified for archiving.")
            if "onnx" not in self._exported_models:
                logger.info("Exporting model to ONNX...")
                self.export(ignore_missing_weights=True)
            path = self._exported_models["onnx"]

        path = Path(path)

        executable_fname = path.name
        archive_name += path.suffix

        mean, scale, color_space = get_preprocessing(
            self.cfg_preprocessing, "Exporting to NN Archive"
        )

        # TODO: keep preprocessing same for each input?
        preprocessing = {
            "mean": mean,
            "scale": scale,
            "dai_type": f"{color_space}888p",
        }

        inputs_dict = get_inputs(path)
        for input_name, metadata in inputs_dict.items():
            inputs.append(
                {
                    "name": input_name,
                    "dtype": metadata["dtype"],
                    "shape": metadata["shape"],
                    "preprocessing": preprocessing,
                    "input_type": "image",
                }
            )

        outputs_dict = get_outputs(path)
        for output_name, metadata in outputs_dict.items():
            outputs.append(
                {
                    "name": output_name,
                    "dtype": metadata["dtype"],
                    "shape": metadata["shape"],
                }
            )

        heads = get_head_configs(self.lightning_module, outputs)

        model = {
            "metadata": {
                "name": self.cfg.model.name,
                "path": executable_fname,
            },
            "inputs": inputs,
            "outputs": outputs,
            "heads": heads,
        }

        cfg_dict = {
            "config_version": CONFIG_VERSION,
            "model": model,
        }

        archive_path = ArchiveGenerator(
            archive_name=archive_name,
            save_path=str(archive_save_directory),
            cfg_dict=cfg_dict,
            executables_paths=[str(path)],  # TODO: what if more executables?
        ).make_archive()

        logger.info(f"NN Archive saved to {archive_path}")

        if self.cfg.archiver.upload_url is not None:  # pragma: no cover
            LuxonisFileSystem.upload(
                archive_path, self.cfg.archiver.upload_url
            )

        if self.cfg.archiver.upload_to_run:
            self.tracker.upload_artifact(archive_path, typ="archive")

        return Path(archive_path)

    @rank_zero_only
    def get_min_loss_checkpoint_path(self) -> str | None:
        """Return best checkpoint path with respect to minimal
        validation loss.

        @rtype: str
        @return: Path to the best checkpoint with respect to minimal
            validation loss
        """
        for callback in self.pl_trainer.checkpoint_callbacks:
            if not isinstance(callback, ModelCheckpoint):
                continue
            if callback.monitor == "val/loss":
                return callback.best_model_path
        return None

    @rank_zero_only
    def get_best_metric_checkpoint_path(self) -> str | None:
        """Return best checkpoint path with respect to best validation
        metric.

        @rtype: str
        @return: Path to the best checkpoint with respect to best
            validation metric
        """
        for callback in self.pl_trainer.checkpoint_callbacks:
            if not isinstance(callback, ModelCheckpoint):
                continue
            if callback.monitor and "val/metric/" in callback.monitor:
                return callback.best_model_path
        return None
