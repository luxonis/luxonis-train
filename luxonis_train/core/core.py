import json
import tempfile
import threading
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from threading import ExceptHookArgs, Thread
from typing import Any, Literal, cast, overload

import lightning.pytorch as pl
import lightning_utilities.core.rank_zero as rank_zero_module
import rich.traceback
import torch
import torch.utils.data as torch_data
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.utilities import rank_zero_only
from loguru import logger
from luxonis_ml.data import LuxonisDataset
from luxonis_ml.data.utils.cli_utils import print_info
from luxonis_ml.nn_archive import ArchiveGenerator
from luxonis_ml.nn_archive.config import CONFIG_VERSION
from luxonis_ml.typing import Params, PathType
from luxonis_ml.utils import Environ, LuxonisFileSystem
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data.dataloader import DataLoader
from typeguard import typechecked

from luxonis_train.callbacks import (
    FailOnNoTrainBatches,
    GracefulInterruptCallback,
    LuxonisRichProgressBar,
    LuxonisTQDMProgressBar,
)
from luxonis_train.config import Config
from luxonis_train.lightning import LuxonisLightningModule
from luxonis_train.lightning.utils import get_main_metric
from luxonis_train.loaders import (
    BaseLoaderTorch,
    DummyLoader,
    LuxonisLoaderTorch,
)
from luxonis_train.loaders.base_loader import LuxonisLoaderTorchOutput
from luxonis_train.registry import (
    LOADERS,
    OPTIMIZERS,
    SCHEDULERS,
    from_registry,
)
from luxonis_train.typing import View
from luxonis_train.utils import (
    DatasetMetadata,
    LuxonisTrackerPL,
    get_tracker_init_params,
    setup_logging,
)
from luxonis_train.utils.general import safe_download

from .utils.annotate_utils import annotate_from_directory
from .utils.archive_utils import (
    get_head_configs,
    get_inputs,
    get_outputs,
)
from .utils.export_utils import (
    blobconverter_export,
    get_preprocessing,
    hubai_export,
    make_initializers_unique,
    rename_onnx_outputs,
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


class LuxonisModel:
    """Common logic of the core components.

    This class contains common logic of the core components (trainer,
    evaluator, exporter, etc.).
    """

    def __init__(
        self,
        cfg: PathType | Params | Config | None,
        opts: Params | list[str] | tuple[str, ...] | None = None,
        *,
        weights: PathType | dict[str, Any] | None = None,
        allow_empty_dataset: bool = False,
        dataset_metadata: DatasetMetadata | None = None,
    ):
        """Construct a new Core instance.

        Loads the config and initializes loaders, dataloaders,
        augmentations, lightning components, etc.

        @type cfg: str | dict[str, Any] | Config
        @param cfg: Path to config file or config dict used to setup
            training.
        @type opts: list[str] | tuple[str, ...] | dict[str, Any] | None
        @param opts: Argument dict provided through command line, used
            for config overriding.
        @type allow_empty_dataset: bool
        @param allow_empty_dataset: If set to True, the model will be
            initialized even if the dataset is empty or cannot be
            created. This is useful either for debugging or for running
            commands that don't require a dataset (e.g. export with
            existing weights).
        @type weights: str | None
        @param weights: Path to the weights. If user specifies weights
            in the config file, the weights provided here will take
            precedence.
        """
        if weights is not None:
            if isinstance(weights, dict):
                if "state_dict" not in weights:
                    weights = {"state_dict": weights}
                ckpt = weights
            elif isinstance(weights, PathType):
                weights = safe_download(weights)
                if weights is None:
                    raise RuntimeError(
                        f"Failed to download weights from {weights}."
                    )
                ckpt = torch.load(weights, map_location="cpu")  # nosemgre
            else:  # pragma: no cover
                raise ValueError(
                    f"Invalid type for weights: {type(weights)}. "
                    "Expected str or dict."
                )

            if cfg is None:
                cfg = ckpt.get("config")
                if cfg is None:
                    raise ValueError(
                        "Checkpoint does not contain the 'config' key. "
                        "Cannot restore `LuxonisModel` from checkpoint."
                    )
            if "dataset_metadata" in ckpt and dataset_metadata is None:
                try:
                    dataset_metadata = DatasetMetadata(
                        **ckpt["dataset_metadata"]
                    )
                except Exception as e:  # pragma: no cover
                    logger.error(
                        "Failed to load dataset metadata from the checkpoint. "
                        f"Error: {e}"
                    )

        if isinstance(cfg, Config):
            self.cfg = cfg
        else:
            self.cfg = Config.get_config(cfg, opts)

        self.allow_empty_dataset = allow_empty_dataset
        self._weights_provided_during_init = weights is not None
        self._weights_provided_in_config = self.cfg.model.weights is not None
        self.weights = weights or self.cfg.model.weights

        self.cfg_preprocessing = self.cfg.trainer.preprocessing

        rich.traceback.install(suppress=[pl, torch], show_locals=False)

        self.tracker = LuxonisTrackerPL(
            rank=rank_zero_only.rank,
            mlflow_tracking_uri=self.environ.MLFLOW_TRACKING_URI,
            _auto_finalize=False,
            **get_tracker_init_params(self.cfg.tracker),
        )

        self.run_save_dir = (
            self.cfg.tracker.save_directory / self.tracker.run_name
        )
        self.log_file = self.run_save_dir / "luxonis_train.log"
        self.error_message = None

        setup_logging(file=self.log_file, use_rich=self.cfg.rich_logging)

        # NOTE: overriding logger in pl so it uses our logger to log device info
        rank_zero_module.log = logger

        if self.cfg.trainer.seed is not None:
            pl.seed_everything(self.cfg.trainer.seed, workers=True)

        self.pl_trainer = create_trainer(
            self.cfg.trainer,
            logger=self.tracker,
            callbacks=[
                GracefulInterruptCallback(self.run_save_dir, self.tracker),
                FailOnNoTrainBatches(),
                LuxonisRichProgressBar()
                if self.cfg.rich_logging
                else LuxonisTQDMProgressBar(),
            ],
            precision=self.cfg.trainer.precision,
        )

        self.loaders: dict[View, BaseLoaderTorch] = {}
        loader_name = self.cfg.loader.name
        Loader = LOADERS.get(loader_name)
        if issubclass(Loader, LuxonisLoaderTorch):
            model_tasks = {
                node.task_name for node in self.cfg.model.head_nodes
            }
            if model_tasks and None not in model_tasks:
                logger.info(
                    f"Using {model_tasks} to filter task names from the dataset"
                )
                self.cfg.loader.params["filter_task_names"] = sorted(
                    model_tasks  # type: ignore
                )

        for view in ("train", "val", "test"):
            if (
                view != "train"
                and issubclass(Loader, LuxonisLoaderTorch)
                and self.cfg.loader.params.get("dataset_dir") is not None
            ):
                self.cfg.loader.params["delete_existing"] = False

            try:
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
                    seed=self.cfg.trainer.seed,
                    **self.cfg.loader.params,  # type: ignore
                )
            except Exception:
                if not self.allow_empty_dataset:
                    logger.error(
                        "Unable to initialize loader. If you want to run "
                        "the model without an existing dataset, "
                        "set `allow_empty_dataset=True`."
                    )
                    raise
                logger.warning(
                    f"Failed to initialize loader '{loader_name}' "
                    f"for view '{view}'. Using `DummyLoader` instead."
                )
                self.cfg.loader.name = DummyLoader.__name__
                self.loaders[view] = DummyLoader(
                    cfg=self.cfg,
                    view={
                        "train": self.cfg.loader.train_view,
                        "val": self.cfg.loader.val_view,
                        "test": self.cfg.loader.test_view,
                    }[view],
                    image_source=self.cfg.loader.image_source,
                    height=self.cfg_preprocessing.train_image_size.height,
                    width=self.cfg_preprocessing.train_image_size.width,
                    color_space=self.cfg_preprocessing.color_space,
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

        self.pytorch_loaders: dict[
            View,
            torch_data.DataLoader[LuxonisLoaderTorchOutput],
        ] = {}
        for view in ("train", "val", "test"):
            if self.cfg.trainer.n_validation_batches is not None and view in {
                "val",
                "test",
            }:
                generator = torch.Generator()
                generator.manual_seed(self.cfg.trainer.seed or 42)
                if self.cfg.trainer.n_validation_batches == -1:
                    loader = self.loaders[view]
                else:
                    n_samples = (
                        self.cfg.trainer.n_validation_batches
                        * self.cfg.trainer.batch_size
                    )
                    subset_size = min(n_samples, len(self.loaders[view]))
                    if subset_size < len(self.loaders[view]):
                        logger.warning(
                            f"Limiting {view} evaluation to the first "
                            f"{subset_size} / {len(self.loaders[view])} "
                            f"samples because "
                            f"`n_validation_batches="
                            f"{self.cfg.trainer.n_validation_batches}`."
                        )
                    indices = range(subset_size)
                    loader = torch_data.Subset(self.loaders[view], indices)
            else:
                loader = self.loaders[view]

            self.pytorch_loaders[view] = torch_data.DataLoader(
                loader,
                batch_size=self.cfg.trainer.batch_size,
                num_workers=self.cfg.trainer.n_workers,
                collate_fn=self.loaders[view].collate_fn,
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

        if dataset_metadata is not None:
            if isinstance(self.loaders["train"], DummyLoader) and set(
                self.cfg.loader.params.keys()
            ).intersection(
                {
                    "class_names",
                    "n_classes",
                    "n_keypoints",
                }
            ):
                logger.warning(
                    "Dataset metadata from the checkpoint are "
                    "overridden by extra loader parameters. "
                    "The checkpoint metadata will not be used."
                )
                self.dataset_metadata = DatasetMetadata.from_loader(
                    self.loaders["train"]
                )
            else:
                self.dataset_metadata = dataset_metadata
        else:
            self.dataset_metadata = DatasetMetadata.from_loader(
                self.loaders["train"]
            )
        logger.info(f"Dataset metadata: {self.dataset_metadata}")
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

        if weights is not None:
            if isinstance(weights, dict):
                if "state_dict" not in weights:
                    weights = {"state_dict": weights}
                ckpt = weights
            else:
                ckpt = LuxonisFileSystem.download(
                    str(weights), self.run_save_dir
                )
            if self.cfg.model.weights is not None:
                logger.warning(
                    "Weights provided in the command line, but config weights are set. "
                    "Ignoring weights provided in config."
                )
            self.lightning_module.load_checkpoint(ckpt)
        self._exported_models: dict[str, Path] = {}

    @property
    def train_loader(self) -> DataLoader:
        return self.pytorch_loaders["train"]

    @property
    def val_loader(self) -> DataLoader:
        return self.pytorch_loaders["val"]

    @property
    def test_loader(self) -> DataLoader:
        return self.pytorch_loaders["test"]

    def save_checkpoint(
        self,
        path: PathType,
        weights_only: bool = False,
        storage_options: Any = None,
    ) -> Path:
        """Save a checkpoint of the model.

        @type path: PathType
        @param path: Path where checkpoint will be saved.
        @type weights_only: bool | None
        @param weights_only: If `True`, will only save the model
            weights.
        @type storage_options: Any
        @param storage_options: parameter for how to save to storage,
            passed to `CheckpointIO` plugin
        @rtype: Path
        @return: Path to the saved checkpoint.
        @raises AttributeError: If the module is not attached to the
            trainer yet. This can happen if you try to save a checkpoint
            before training / evaluating the model first.
        """
        self.pl_trainer.save_checkpoint(
            path, weights_only=weights_only, storage_options=storage_options
        )
        return Path(path)

    def get_checkpoint(self, weights_only: bool = False) -> dict[str, Any]:
        """Get the checkpoint of the model as a dictionary.

        @type weights_only: bool
        @param weights_only: If `True`, will only include the model
            weights in the checkpoint.
        @rtype: dict[str, Any]
        @return: Checkpoint of the model as a dictionary.
        @raises AttributeError: If the module is not attached to the
            trainer yet. This can happen if you try to save a checkpoint
            before training / evaluating the model first.
        """
        with tempfile.NamedTemporaryFile(suffix=".ckpt", delete=False) as tmp:
            checkpoint_path = self.save_checkpoint(tmp.name, weights_only)
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            checkpoint_path.unlink(missing_ok=True)
        return ckpt

    def _train(self, resume: PathType | None, *args, **kwargs) -> None:
        status = "success"
        try:
            self.pl_trainer.fit(*args, ckpt_path=resume, **kwargs)
        except Exception:  # pragma: no cover
            logger.exception("Encountered an exception during training.")
            status = "failed"
            raise
        finally:
            self.finalize_run(status)

    def train(
        self,
        new_thread: bool = False,
        weights: PathType | None = None,
    ) -> None:
        """Run the training.

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

        weights = self.resolve_weights(weights)  # type: ignore

        if self.cfg.trainer.resume_training and weights is None:
            logger.warning(
                "Resume training is enabled but no weights were provided. "
                "Training will start from scratch."
            )
        elif weights and self.cfg.trainer.resume_training is False:
            logger.info(
                "Weights argument was given and resume_training is set to False. "
                "Training will start from the provided weights while resetting "
                "optimizer, scheduler, and epoch state."
            )
            self.lightning_module.load_checkpoint(weights)

        resume_weights = weights if self.cfg.trainer.resume_training else None

        if not new_thread:
            logger.info(f"Checkpoints will be saved in: {self.run_save_dir}")
            logger.info("Starting training...")
            self._train(
                resume_weights,
                self.lightning_module,
                self.train_loader,
                self.val_loader,
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
                    self.train_loader,
                    self.val_loader,
                ),
                daemon=True,
            )
            self.thread.start()

    def export(
        self,
        save_path: PathType | None = None,
        weights: PathType | dict[str, Any] | None = None,
        ignore_missing_weights: bool = False,
        ckpt_only: bool = False,
    ) -> Path:
        """Run the export.

        @type save_path: PathType | None
        @param save_path: Directory where to save all exported model
            files. If not specified, files will be saved to the "export"
            directory in the run save directory.
        @type weights: PathType | None
        @param weights: Path to the checkpoint from which to load
            weights. If not specified, the value of `model.weights` from
            the configuration file will be used. The current weights of
            the model will be temporarily replaced with the weights from
            the specified checkpoint.
        @type ignore_missing_weights: bool
        @param ignore_missing_weights: If set to True, the warning about
            missing weights will be suppressed.
        @type ckpt_only: bool
        @param ckpt_only: If True, only the `.ckpt` file will be
            exported. This is useful for updating the metadata in the
            checkpoint file in case they changed (e.g. new configuration
            file, architectural changes affecting the exection order
            etc.)
        @rtype: Path
        @return: Path to the exported ONNX model file or .ckpt file if
            ckpt_only is True.
        """
        weights = self.resolve_weights(weights)

        if not ignore_missing_weights and weights is None:
            logger.warning(
                "No model weights specified. Exporting model without weights."
            )
        if save_path is None:
            export_path = (
                self.run_save_dir
                / "export"
                / (self.cfg.exporter.name or self.cfg.model.name)
            )
        else:
            save_path = Path(save_path)
            if save_path.suffix:
                export_path = save_path.with_suffix("")
            else:
                export_path = save_path / (
                    self.cfg.exporter.name or self.cfg.model.name
                )
        export_path.parent.mkdir(parents=True, exist_ok=True)

        if ckpt_only:
            logger.info("Re-exporting the checkpoint file.")
            with replace_weights(self.lightning_module, weights):
                # Needs to be called to attach the model to the trainer
                self.pl_trainer.strategy._lightning_module = (
                    self.lightning_module
                )
                self.pl_trainer.save_checkpoint(
                    str(export_path.with_suffix(".ckpt")), weights_only=False
                )
                logger.info(
                    f"Checkpoint saved to {export_path.with_suffix('.ckpt')}"
                )
            return export_path.with_suffix(".ckpt")

        with replace_weights(self.lightning_module, weights):
            onnx_kwargs = self.cfg.exporter.onnx.model_dump(
                exclude={
                    "disable_onnx_simplification",
                    "unique_onnx_initializers",
                }
            )
            onnx_save_path = self.lightning_module.export_onnx(
                export_path.with_suffix(".onnx"), **onnx_kwargs
            )

        if not self.cfg.exporter.onnx.disable_onnx_simplification:
            try_onnx_simplify(onnx_save_path)

        if self.cfg.exporter.onnx.unique_onnx_initializers:
            make_initializers_unique(onnx_save_path)

        self._exported_models["onnx"] = Path(onnx_save_path)

        mean, scale, color_space = get_preprocessing(
            self.cfg_preprocessing, "Model export"
        )
        scale_values = self.cfg.exporter.scale_values or scale
        mean_values = self.cfg.exporter.mean_values or mean

        for path in self._exported_models.values():
            if self.cfg.exporter.upload_to_run:
                self.tracker.upload_artifact(path, typ="export")
            if self.cfg.exporter.upload_url is not None:  # pragma: no cover
                LuxonisFileSystem.upload(path, self.cfg.exporter.upload_url)

        if len(self.input_shapes) > 1:
            logger.error(
                "Generating modelconverter config for a model "
                "with multiple inputs is not implemented yet."
            )
            return onnx_save_path

        inputs = []
        outputs = []
        inputs_dict = get_inputs(self._exported_models["onnx"])
        for input_name, metadata in inputs_dict.items():
            inputs.append(
                {
                    "name": input_name,
                    "shape": metadata["shape"],
                }
            )

        outputs_dict = get_outputs(self._exported_models["onnx"])
        for output_name, metadata in outputs_dict.items():
            outputs.append(
                {
                    "name": output_name,
                    "shape": metadata["shape"],
                }
            )
        modelconverter_config = {
            "input_model": onnx_save_path,
            "scale_values": scale_values,
            "mean_values": mean_values,
            "encoding": {"from": color_space, "to": "BGR"},
            "inputs": inputs,
            "outputs": outputs,
        }

        with open(export_path.with_suffix(".yaml"), "w") as f:
            yaml.safe_dump(
                modelconverter_config,
                f,
                sort_keys=False,
                default_flow_style=False,
            )
            if self.cfg.exporter.upload_to_run:
                self.tracker.upload_artifact(f.name, name=f.name, typ="export")
            if self.cfg.exporter.upload_url is not None:  # pragma: no cover
                LuxonisFileSystem.upload(f.name, self.cfg.exporter.upload_url)

        return onnx_save_path

    @overload
    def test(
        self,
        new_thread: Literal[False] = ...,
        view: Literal["train", "test", "val"] = "test",
        weights: PathType | dict[str, Any] | None = ...,
        finalize_tracker: bool = True,
    ) -> Mapping[str, float]: ...

    @overload
    def test(
        self,
        new_thread: Literal[True] = ...,
        view: Literal["train", "test", "val"] = "test",
        weights: PathType | dict[str, Any] | None = ...,
        finalize_tracker: bool = True,
    ) -> Thread: ...

    @typechecked
    def test(
        self,
        new_thread: bool = False,
        view: Literal["train", "val", "test"] = "test",
        weights: PathType | dict[str, Any] | None = None,
        finalize_tracker: bool = True,
    ) -> Mapping[str, float] | Thread:
        """Runs testing.

        @type new_thread: bool
        @param new_thread: Runs testing in a new thread if set to True.
        @type view: Literal["train", "test", "val"]
        @param view: Which view to run the testing on. Defauls to
            "test".
        @type weights: PathType | None
        @param weights: Path to the checkpoint from which to load
            weights. If not specified, the value of `model.weights` from
            the configuration file will be used. The current weights of
            the model will be temporarily replaced with the weights from
            the specified checkpoint.
        @type finalize_tracker: bool
        @param finalize_tracker: If True, uploads final run metadata and
            finalizes the tracker after testing. Set to False only when
            the current run is expected to continue with additional
            actions such as export or archive, in which case the caller
            is responsible for eventually calling L{finalize_run()}.
        @rtype: Mapping[str, float] | Thread
        @return: If new_thread is False, returns a dictionary test
            results.
        """
        weights = self.resolve_weights(weights)
        loader = self.pytorch_loaders[view]

        def _run_test() -> Mapping[str, float]:
            status = "success"
            try:
                with replace_weights(self.lightning_module, weights):
                    return self.pl_trainer.test(self.lightning_module, loader)[
                        0
                    ]
            except Exception:  # pragma: no cover
                logger.exception("Encountered an exception during testing.")
                status = "failed"
                raise
            finally:
                if finalize_tracker:
                    self.finalize_run(status)

        if new_thread:  # pragma: no cover
            self.thread = threading.Thread(
                target=_run_test,
                daemon=True,
            )
            self.thread.start()
            return self.thread
        return _run_test()

    def finalize_run(self, status: str = "success") -> None:
        """Upload run metadata and finalize the tracker.

        @type status: str
        @param status: Final run status passed to the tracker.
        """
        self._upload_run_metadata()
        self.tracker._finalize(status)

    def _upload_run_metadata(self) -> None:
        self.tracker.upload_artifact(self.log_file, typ="logs")
        self.tracker.upload_artifact(self.config_file, typ="config")

    def infer(
        self,
        view: Literal["train", "val", "test"] = "val",
        save_dir: PathType | None = None,
        source_path: PathType | None = None,
        weights: PathType | dict[str, Any] | None = None,
    ) -> None:
        """Run the inference.

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
        weights = self.resolve_weights(weights)

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

    def annotate(
        self,
        dir_path: PathType,
        dataset_name: str,
        weights: PathType | dict[str, Any] | None = None,
        bucket_storage: Literal["local", "gcs"] = "local",
        delete_local: bool = True,
        delete_remote: bool = True,
        team_id: str | None = None,
    ) -> LuxonisDataset:
        self.lightning_module.eval()
        weights = self.resolve_weights(weights)

        with replace_weights(self.lightning_module, weights):
            dir_path = Path(dir_path)
            if dir_path.is_dir():
                image_files = (
                    f
                    for f in dir_path.iterdir()
                    if f.suffix.lower() in IMAGE_FORMATS
                )
                annotated_dataset = annotate_from_directory(
                    self,
                    image_files,
                    dataset_name,
                    bucket_storage,
                    delete_local,
                    delete_remote,
                    team_id,
                )
            else:
                raise ValueError(
                    f"Directory path {dir_path} is not a valid directory."
                )

        print_info(annotated_dataset)

        return annotated_dataset

    def tune(self) -> None:
        """Run Optuna tuning of hyperparameters."""
        import optuna
        from optuna.integration import PyTorchLightningPruningCallback
        from sqlalchemy import URL

        from .utils.tune_utils import (
            get_trial_params,
            rename_params_for_logging,
        )

        def _objective(trial: optuna.trial.Trial) -> float:
            """Objective function used to optimize Optuna study."""
            cfg_tracker = self.cfg.tracker
            tracker_params = get_tracker_init_params(cfg_tracker)
            tracker_params["run_name"] = (
                tracker_params["run_name"] or self.tracker.run_name
            )
            child_tracker = LuxonisTrackerPL(
                rank=rank_zero_only.rank,
                mlflow_tracking_uri=self.environ.MLFLOW_TRACKING_URI,
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

            renamed_params = rename_params_for_logging(
                curr_params, self.cfg.tuner.params
            )
            child_tracker.log_hyperparams(renamed_params)

            cfg.save_data(run_save_dir / "training_config.yaml")
            cfg.trainer.n_sanity_val_steps = 0
            lightning_module = LuxonisLightningModule(
                cfg=cfg,
                dataset_metadata=self.dataset_metadata,
                save_dir=run_save_dir,
                input_shapes=self.loaders["train"].input_shapes,
                _core=self,
            )
            callbacks: list[pl.Callback] = [
                (
                    LuxonisRichProgressBar()
                    if cfg.rich_logging
                    else LuxonisTQDMProgressBar()
                )
            ]

            if cfg.tuner.monitor == "loss":
                monitor = "val/loss"
            else:
                main_metric = get_main_metric(cfg)
                if main_metric is None:  # pragma: no cover
                    raise ValueError(
                        "You have to specify the `main_metric` in the `model.metrics` section of the config when using a custom metric for tuning."
                    )
                all_mlflow_logging_keys = self.get_mlflow_logging_keys()
                search_name = (
                    "mcc"
                    if main_metric.metric_name == "ConfusionMatrix"
                    else main_metric.metric_name
                )
                monitor = next(
                    (
                        k
                        for k in all_mlflow_logging_keys["metrics"]
                        if search_name in k
                        and main_metric.node_name in k
                        and "val" in k
                    ),
                    None,
                )
                if monitor is None:
                    raise ValueError(
                        f"Could not find monitor key for main metric '{main_metric.metric_name}' "
                        f"attached to '{main_metric.node_name}' in the MLFlow logging keys."
                    )

            pruner_callback = PyTorchLightningPruningCallback(
                trial, monitor=monitor
            )
            graceful_interrupt_callback = GracefulInterruptCallback(
                self.run_save_dir, self.tracker
            )
            fail_no_train_batches_callback = FailOnNoTrainBatches()

            callbacks.append(pruner_callback)
            callbacks.append(graceful_interrupt_callback)
            callbacks.append(fail_no_train_batches_callback)

            if self.cfg.trainer.seed is not None:
                pl.seed_everything(cfg.trainer.seed, workers=True)

            pl_trainer = create_trainer(
                cfg.trainer, logger=child_tracker, callbacks=callbacks
            )

            try:
                pl_trainer.fit(
                    lightning_module,
                    self.train_loader,
                    self.val_loader,
                )
                pruner_callback.check_pruned()

            # Pruning is done by raising an error
            except optuna.TrialPruned as e:
                logger.info(e)

            return pl_trainer.callback_metrics[monitor].item()

        cfg_tuner = self.cfg.tuner
        if cfg_tuner is None:
            raise ValueError(
                "You have to specify the `tuner` section in config."
            )

        all_augs = [a.name for a in self.cfg_preprocessing.augmentations]
        rank = rank_zero_only.rank
        cfg_tracker = self.cfg.tracker
        tracker_params = get_tracker_init_params(cfg_tracker)
        # NOTE: wandb doesn't allow multiple concurrent runs, handle this separately
        tracker_params["is_wandb"] = False
        tracker_params["run_name"] = (
            tracker_params["run_name"] or self.tracker.run_name
        )
        self.parent_tracker = LuxonisTrackerPL(
            rank=rank,
            mlflow_tracking_uri=self.environ.MLFLOW_TRACKING_URI,
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

        if cfg_tuner.storage.active:
            storage = URL.create(
                cfg_tuner.storage.backend,
                username=cfg_tuner.storage.username,
                password=cfg_tuner.storage.password.get_secret_value()
                if cfg_tuner.storage.password is not None
                else None,
                host=cfg_tuner.storage.host,
                database=cfg_tuner.storage.database,
                port=cfg_tuner.storage.port,
            )
            logger.info(f"Using '{storage}' as Optuna storage.")
        else:
            storage = None

        study = optuna.create_study(
            study_name=cfg_tuner.study_name,
            storage=storage.render_as_string(hide_password=False)
            if storage
            else None,
            direction="minimize"
            if cfg_tuner.monitor == "loss"
            else "maximize",
            pruner=pruner,
            load_if_exists=cfg_tuner.continue_existing_study,
        )

        study.optimize(
            _objective, n_trials=cfg_tuner.n_trials, timeout=cfg_tuner.timeout
        )
        logger.info(
            f"Best study parameters: {study.best_params}. Cost: {study.best_value}."
        )

        study_df = study.trials_dataframe()
        study_df.to_csv(self.run_save_dir / "tuner_study.csv", index=False)

        logger.info(
            f"Optuna study results saved to {self.run_save_dir / 'tuner_study.csv'}."
        )

        self.parent_tracker.log_hyperparams(study.best_params)

        if self.cfg.tracker.is_wandb:  # pragma: no cover
            # If wandb used then init parent tracker separately at the end
            wandb_parent_tracker = LuxonisTrackerPL(
                rank=rank_zero_only.rank,
                _auto_finalize=True,
                **(
                    get_tracker_init_params(self.cfg.tracker)
                    | {"run_name": self.parent_tracker.run_name}
                ),
            )
            wandb_parent_tracker.log_hyperparams(study.best_params)

    def archive(
        self,
        path: PathType | None = None,
        weights: PathType | dict[str, Any] | None = None,
        save_dir: PathType | None = None,
    ) -> Path:
        """Generate an NN Archive out of a model executable.

        @type path: PathType | None
        @param path: Path to the model executable. If not specified, the
            model will be exported first.
        @type weights: PathType | None
        @param weights: Path to the checkpoint from which to load
            weights. If not specified, the value of `model.weights` from
            the configuration file will be used. The current weights of
            the model will be temporarily replaced with the weights from
            the specified checkpoint.
        @rtype: Path
        @return: Path to the generated NN Archive.
        """
        weights = self.resolve_weights(weights)
        with replace_weights(self.lightning_module, weights):
            return self._archive(path, save_dir)

    def _archive(
        self, path: PathType | None = None, save_dir: PathType | None = None
    ) -> Path:
        if isinstance(save_dir, str):
            save_dir = Path(save_dir)

        archive_name = self.cfg.archiver.name or self.cfg.model.name
        archive_save_directory = save_dir or Path(self.run_save_dir, "archive")
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
        executable_paths: list[PathType] = [path]

        external_data_path = path.with_name(f"{path.name}.data")
        if external_data_path.exists():
            executable_paths.append(external_data_path)

        executable_fname = path.name
        archive_name += path.suffix

        mean, scale, color_space = get_preprocessing(
            self.cfg_preprocessing, "Exporting to NN Archive"
        )
        scale_values = self.cfg.exporter.scale_values or scale
        mean_values = self.cfg.exporter.mean_values or mean

        # TODO: keep preprocessing same for each input?
        preprocessing = {
            "mean": mean_values,
            "scale": scale_values,
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
            executables_paths=executable_paths,
        ).make_archive()

        logger.info(f"NN Archive saved to {archive_path}")

        if self.cfg.archiver.upload_url is not None:  # pragma: no cover
            LuxonisFileSystem.upload(
                archive_path, self.cfg.archiver.upload_url
            )

        if self.cfg.archiver.upload_to_run:
            self.tracker.upload_artifact(archive_path, typ="archive")

        return Path(archive_path)

    def convert(
        self,
        weights: PathType | dict[str, Any] | None = None,
        save_dir: PathType | None = None,
    ) -> tuple[Path, dict[str, Path]]:
        """Export the model to ONNX, creates an NN Archive, and converts
        to target platform format (RVC2/RVC3/RVC4).

        This is a unified method that combines export, archive, and
        platform conversion steps.

        @type weights: PathType | None
        @param weights: Path to the checkpoint from which to load
            weights. If not specified, the value of `model.weights` from
            the configuration file will be used.
        @type save_dir: PathType | None
        @param save_dir: Directory where the outputs will be saved. If
            not specified, the default run save directory will be used.
        @rtype: tuple[Path, dict[str, Path]]
        @return: A tuple of: 1) Path to the generated ONNX-based NN
            Archive. 2) Mapping of additional conversion artifact names
            to their paths.
        """
        self.export(weights=weights, save_path=save_dir)

        onnx_path = self._exported_models.get("onnx")
        if onnx_path is None:
            raise RuntimeError(
                "ONNX export failed, cannot proceed with conversion."
            )

        archive_path = self.archive(
            path=onnx_path, weights=weights, save_dir=save_dir
        )

        mean, scale, color_space = get_preprocessing(
            self.cfg_preprocessing, "Model conversion"
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

        convert_save_dir = (
            Path(save_dir) if save_dir else Path(self.run_save_dir)
        )

        conversion_artifacts: dict[str, Path] = {}
        if self.cfg.exporter.blobconverter.active:
            logger.warning(
                "blobconverter is deprecated and only supports RVC2 legacy conversion to `.blob`. "
                "Please consider using the HubAI SDK instead."
            )
            try:
                blob_path = blobconverter_export(
                    self.cfg.exporter,
                    scale_values,
                    mean_values,
                    reverse_input_channels,
                    str(convert_save_dir),
                    str(onnx_path),
                )
                blob_path = Path(blob_path)
                self._exported_models["blob"] = blob_path
                conversion_artifacts["blob"] = blob_path
                if self.cfg.exporter.upload_to_run:
                    self.tracker.upload_artifact(blob_path, typ="export")
                if self.cfg.exporter.upload_url is not None:
                    LuxonisFileSystem.upload(
                        blob_path, self.cfg.exporter.upload_url
                    )
            except ImportError:
                logger.error("Failed to import `blobconverter`")
                logger.warning(
                    "`blobconverter` not installed. Skipping .blob model conversion. "
                    "Ensure `blobconverter` is installed in your environment."
                )

        if self.cfg.exporter.hubai.active:
            try:
                dataset_name = None
                if "train" in self.loaders and hasattr(
                    self.loaders["train"], "dataset"
                ):
                    dataset = getattr(self.loaders["train"], "dataset", None)
                    if dataset is not None:
                        dataset_name = getattr(dataset, "dataset_name", None)
                hubai_archive_path = Path(
                    hubai_export(
                        cfg=self.cfg.exporter.hubai,
                        quantization_mode=self.cfg.exporter.quantization_mode,
                        archive_path=archive_path,
                        export_path=convert_save_dir,
                        model_name=self.cfg.model.name,
                        dataset_name=dataset_name,
                    )
                )
                conversion_artifacts["hubai_archive"] = hubai_archive_path
                self._exported_models["hubai_archive"] = hubai_archive_path
                if self.cfg.archiver.upload_to_run:
                    self.tracker.upload_artifact(
                        hubai_archive_path, typ="archive"
                    )
                if self.cfg.archiver.upload_url is not None:
                    LuxonisFileSystem.upload(
                        hubai_archive_path, self.cfg.archiver.upload_url
                    )
            except ImportError:
                logger.error("Failed to import `hubai_sdk`")
                logger.warning(
                    "`hubai-sdk` not installed. Skipping HubAI conversion. "
                    "Ensure `hubai-sdk` is installed in your environment."
                )
            except ValueError as e:
                raise ValueError(f"HubAI conversion failed: {e}") from e

        return archive_path, conversion_artifacts

    def quantize(
        self,
        weights: PathType | None = None,
        epochs: int | None = None,
        quant_scheme: Literal["min_max", "tf", "tf_enhanced"] | None = None,
        default_output_bw: int | None = None,
        default_param_bw: int | None = None,
        config_file: str | None = None,
        default_data_type: Literal["int", "float"] | None = None,
        adaround: bool | None = None,
        adaround_iterations: int | None = None,
        adaround_reg_param: float | None = None,
        adaround_beta_range: tuple[int, int] | None = None,
        adaround_warm_start: float | None = None,
        fold_batch_norms: bool | None = None,
        cross_layer_equalization: bool | None = None,
        batch_norm_reestimation: bool | None = None,
        sequential_mse: bool | None = None,
        optimizer: Optimizer | None = None,
        scheduler: LRScheduler | None = None,
        in_place: bool = False,
    ) -> Path:
        """Quantize the model using AIMET.

        @type weights: PathType | None
        @param weights: Path to the checkpoint from which to load
            weights.
        @type epochs: int | None
        @param epochs: Number of epochs to run quantization-aware
            training for.
        @type quant_scheme: str | QuantScheme | None
        @param quant_scheme: Quantization scheme to use. If not
            specified, the value from the configuration file will be
            used.
        @type default_output_bw: int | None
        @param default_output_bw: Default bitwidth to use for quantizing
            outputs. If not specified, the value from the configuration
            file will be used.
        @type default_param_bw: int | None
        @param default_param_bw: Default bitwidth to use for quantizing
            parameters. If not specified, the value from the
            configuration file will be used.
        @type config_file: str | None
        @param config_file: Path to the AIMET configuration file or a
            dictionary containing the AIMET configuration. If not
            specified, the value from the configuration file will be
            used.
        @type default_data_type: QuantizationDataType | None
        @param default_data_type: Default data type to use for
            quantization. If not specified, the value from the
            configuration file will be used.
        @type adaround: bool | None
        @param adaround: Whether to use Adaround for weight
            quantization. If not specified, the value from the
            configuration file will be used.
        @type adaround_iterations: int | None
        @param adaround_iterations: Number of iterations to run Adaround
            for. If not specified, the value from the configuration file
            will be used.
        @type adaround_reg_param: float | None
        @param adaround_reg_param: Regularization parameter to use for
            Adaround. If not specified, the value from the configuration
            file will be used.
        @type adaround_beta_range: tuple[int, int] | None
        @param adaround_beta_range: Beta range to use for Adaround. If
            not specified, the value from the configuration file will be
            used.
        @type adaround_warm_start: float | None = None
        @param adaround_warm_start: Warm start value to use for
            Adaround. If not specified, the value from the configuration
            file will be used.
        @type fold_batch_norms: bool | None
        @param fold_batch_norms: Whether to fold batch norms before
            quantization. If not specified, the value from the
            configuration file will be used.
        @type cross_layer_equalization: bool | None
        @param cross_layer_equalization: Whether to perform cross-layer
            equalization before quantization. If not specified, the
            value from the configuration file will be used.
        @type batch_norm_reestimation: bool | None
        @param batch_norm_reestimation: Whether to perform batch norm
            reestimation after folding batch norms. If not specified,
            the value from the configuration file will be used.
        @type optimizer: Optimizer | None
        @param optimizer: Optimizer to use for quantization-aware
            training. If not specified, the optimizer from the
            configuration file will be used.
        @type scheduler: LRScheduler | None
        @param scheduler: Learning rate scheduler to use for
            quantization-aware training. If not specified, the scheduler
            from the configuration file will be used.
        @type in_place: bool
        @param in_place: Whether to perform quantization in-place on the
            original model or to create a copy of the model for
            quantization. Defaults to False, which means that a copy of
            the model will be created for quantization. Setting this to
            True will modify the original model in-place, which may save
            memory but will overwrite the original model's weights and
            structure.
        """
        from .utils.aimet_utils import (
            check_aimet_available,
            get_ptq_calibration_loader,
            post_training_quantization,
            quantization_aware_training,
        )

        check_aimet_available()

        from aimet_torch.common.defs import (  # pyright: ignore[reportMissingImports]
            QuantizationDataType,
            QuantScheme,
        )

        save_dir = self.run_save_dir / "aimet"
        save_dir.mkdir(parents=True, exist_ok=True)

        cfg = self.cfg.exporter.aimet

        aimet_config_file = config_file or cfg.config
        if isinstance(aimet_config_file, dict):
            with open(save_dir / "aimet_config.json", "w") as f:
                json.dump(aimet_config_file, f, indent=4)
            aimet_config_file = str(save_dir / "aimet_config.json")

        adaround = adaround if adaround is not None else cfg.adaround.active
        fold_batch_norms = (
            fold_batch_norms
            if fold_batch_norms is not None
            else cfg.fold_batch_norms
        )
        cross_layer_equalization = (
            cross_layer_equalization
            if cross_layer_equalization is not None
            else cfg.cross_layer_equalization
        )
        batch_norm_reestimation = (
            batch_norm_reestimation
            if batch_norm_reestimation is not None
            else cfg.batch_norm_reestimation
        )
        sequential_mse = (
            sequential_mse
            if sequential_mse is not None
            else cfg.sequential_mse
        )

        if not in_place:
            model = deepcopy(self.lightning_module)
        else:
            model = self.lightning_module

        model.reparametrize().eval()

        if weights is not None:
            model.load_checkpoint(weights)

        # Lightning test loops use inference mode by default, which can
        # leak inference tensors into lazily initialized loss buffers that
        # are later reused by QAT under autograd.
        quant_eval_trainer = create_trainer(
            self.cfg.trainer,
            logger=self.tracker,
            callbacks=[
                LuxonisRichProgressBar()
                if self.cfg.rich_logging
                else LuxonisTQDMProgressBar()
            ],
            precision=self.cfg.trainer.precision,
            inference_mode=False,
        )

        pre_quant_test = quant_eval_trainer.test(model, self.val_loader)[0]

        dummy_inputs = {
            input_name: torch.randn([1, *shape]).to(model.device)
            for shapes in model.nodes.loader_input_shapes.values()
            for input_name, shape in shapes.items()
        }

        if len(dummy_inputs) > 1:
            raise NotImplementedError(
                "Quantization is not yet supported for models "
                "with multiple inputs."
            )
        input_names = list(dummy_inputs.keys())
        output_names = model._get_output_onnx_names(deepcopy(dummy_inputs))
        dummy_inputs = next(iter(dummy_inputs.values()))
        ptq_loader = get_ptq_calibration_loader(
            val_dataset=self.loaders["val"],
            collate_fn=self.loaders["val"].collate_fn,
            batch_size=self.cfg.trainer.batch_size,
            num_workers=self.cfg.trainer.n_workers,
            pin_memory=self.cfg.trainer.pin_memory,
            max_calibration_images=cfg.max_calibration_images,
        )

        sim = post_training_quantization(
            model,
            dummy_inputs,
            ptq_loader,
            save_dir,
            QuantScheme.from_str(quant_scheme or cfg.quant_scheme),
            default_output_bw or cfg.default_output_bw,
            default_param_bw or cfg.default_param_bw,
            QuantizationDataType[default_data_type or cfg.default_data_type],
            aimet_config_file,
            adaround,
            adaround_iterations
            if adaround_iterations is not None
            else cfg.adaround.default_num_iterations,
            adaround_reg_param
            if adaround_reg_param is not None
            else cfg.adaround.default_reg_param,
            adaround_beta_range or cfg.adaround.default_beta_range,
            adaround_warm_start
            if adaround_warm_start is not None
            else cfg.adaround.default_warm_start,
            fold_batch_norms,
            cross_layer_equalization,
            batch_norm_reestimation,
            sequential_mse,
        )
        model = cast(LuxonisLightningModule, sim.model)

        model.eval()
        ptq_test = quant_eval_trainer.test(model, self.val_loader)[0]

        if optimizer is None:
            optimizer = from_registry(
                OPTIMIZERS,
                cfg.optimizer.name,
                params=sim.model.parameters(),
                **cfg.optimizer.params,
            )
        if scheduler is None:
            scheduler = from_registry(
                SCHEDULERS,
                cfg.scheduler.name,
                optimizer=optimizer,
                **cfg.scheduler.params,
            )

        model = quantization_aware_training(
            sim,
            dummy_inputs,
            self.train_loader,
            optimizer,
            scheduler,
            epochs if epochs is not None else cfg.epochs,
            fold_batch_norms,
            batch_norm_reestimation,
        ).eval()

        qat_test = quant_eval_trainer.test(model, self.val_loader)[0]

        model.set_export_mode(mode=True)

        sim.onnx.export(
            dummy_inputs,
            (save_dir / self.cfg.model.name).with_suffix(".onnx"),
            input_names=input_names,
            output_names=output_names,
        )
        rename_onnx_outputs(
            (save_dir / self.cfg.model.name).with_suffix(".onnx"),
            output_names,
        )
        self._archive(
            path=(save_dir / self.cfg.model.name).with_suffix(".onnx"),
            save_dir=save_dir,
        )

        table = []
        for key, value in pre_quant_test.items():
            log_key = key.replace("test/metric/", "").replace("test/loss/", "")
            table.append((log_key, value, ptq_test[key], qat_test[key]))
        model.progress_bar.print_table(
            "Quantization results",
            table,
            ["Name", "Pre-Quant", "PTQ", "QAT"],
        )
        logger.info(f"AIMET artifacts saved in: {save_dir}")
        return save_dir

    @property
    def environ(self) -> Environ:
        return self.cfg.ENVIRON

    @rank_zero_only
    def get_min_loss_checkpoint_path(self) -> str | None:
        """Get the best checkpoint path with respect to minimal
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
        """Get the best checkpoint path with respect to best validation
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

    def get_mlflow_logging_keys(self) -> dict[str, list[str]]:
        """
        Return a dictionary with two lists of keys:
        1) "metrics"    -> Keys expected to be logged as standard metrics
        2) "artifacts"  -> Keys expected to be logged as artifacts (e.g. confusion_matrix.json, visualizations).
        """
        return self.lightning_module.get_mlflow_logging_keys()

    def resolve_weights(
        self, weights: PathType | dict[str, Any] | None
    ) -> PathType | dict[str, Any] | None:
        if isinstance(weights, dict):
            if "state_dict" not in weights:
                weights = {"state_dict": weights}
            return weights

        if weights is None:
            if isinstance(self.weights, dict):
                return self.weights
            return safe_download(self.weights)

        if (
            self._weights_provided_in_config
            and not self._weights_provided_during_init
        ):
            logger.warning(
                "Weights provided on the command line, but config weights are set. "
                "Ignoring weights provided in config."
            )
        return safe_download(weights)
