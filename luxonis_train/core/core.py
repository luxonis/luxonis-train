"""`LuxonisModel`, the entry point of the package.

It owns the config, the loaders, and the Lightning module. It exposes
the CLI commands that act on a model as methods.

"""

import json
import tempfile
import threading
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from threading import ExceptHookArgs, Thread
from typing import TYPE_CHECKING, Any, Literal, cast, overload

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
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data.dataloader import DataLoader
from typeguard import typechecked

if TYPE_CHECKING:
    import optuna
    from aimet_torch import (  # pyright: ignore[reportMissingImports]
        QuantizationSimModel,
    )
    from sqlalchemy import URL

from luxonis_train.callbacks import (
    FailOnNoTrainBatches,
    GracefulInterruptCallback,
    LuxonisRichProgressBar,
    LuxonisTQDMProgressBar,
)
from luxonis_train.config import Config
from luxonis_train.config.config import (
    AIMETConfig,
    CallbackConfig,
    TunerConfig,
)
from luxonis_train.config.predefined import resolve_predefined_config
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
    """The model of one run, and the commands that act on it.

    One instance holds the config, the tracker, the Lightning trainer,
    a loader for each dataset view, and the `LuxonisLightningModule`
    that runs the node graph. The constructor builds all of them and
    creates the run directory. The public methods map to the commands
    of the ``luxonis_train`` CLI: `train`, `test`, `infer`, `annotate`,
    `export`, `archive`, `convert`, `tune`, and `quantize`.

    Attributes:
        cfg (Config): The config of the run.
        cfg_preprocessing (PreprocessingConfig): Shortcut to
            ``cfg.trainer.preprocessing``.
        tracker (LuxonisTrackerPL): The experiment tracker of the run.
        run_save_dir (``Path``): ``<tracker.save_directory>/<run name>``.
            The checkpoints, the logs, and the exported files go there.
        error_message (str | None): The message of the exception that
            ended a training thread, or ``None``.
        pl_trainer (``pl.Trainer``): The Lightning trainer.
        loaders (``dict[View, BaseLoaderTorch]``): The loaders, keyed by
            ``"train"``, ``"val"``, and ``"test"``.
        pytorch_loaders (``dict[View, DataLoader]``): The PyTorch data
            loaders over ``loaders``, with the same keys.
        lightning_module (LuxonisLightningModule): The module that runs
            the node graph.
        thread (``threading.Thread``): The thread of the last `train` or
            `test` call with ``new_thread=True``. Unset before that.

    Example:
        .. code-block:: python

            from luxonis_train import LuxonisModel

            model = LuxonisModel("config.yaml", opts={"trainer.epochs": 10})
            model.train()
            model.test(view="test")
            model.export()

    """

    def __init__(
        self,
        cfg: PathType | Params | Config | None = None,
        opts: Params | list[str] | tuple[str, ...] | None = None,
        *,
        model: str | None = None,
        variant: str | None = None,
        weights: PathType | dict[str, Any] | None = None,
        allow_empty_dataset: bool = False,
        dataset_metadata: DatasetMetadata | None = None,
    ):
        """Build the run from a config, a checkpoint, or a model name.

        Give ``cfg`` or ``model``, not both. With neither, the config
        comes from the ``config`` key of the ``weights`` checkpoint.
        Without ``weights`` too, ``opts`` alone builds the config from
        the defaults. `Config.get_config` raises ``ValueError`` when
        ``opts`` is ``None`` as well.

        Args:
            cfg (``PathType | Params | Config | None``): A path or URL of a
                config file, a config dictionary, or a `Config` instance.
                ``opts`` do not apply to a `Config` instance.
            opts (``Params | list[str] | tuple[str, ...] | None``): Overrides
                of the config, as a mapping of dotted keys to values or
                as a flat sequence of alternating keys and values.
            model (str | None): The name of a packaged predefined model,
                with an optional version suffix, as in ``"detection:v1"``
                or ``"detection:latest"``. Its packaged YAML file becomes
                ``cfg``. A version other than ``latest`` goes to ``opts``
                as ``model.predefined_model.version``, and ``variant``
                goes there as ``model.predefined_model.variant``.
            variant (str | None): The variant of the packaged model.
                Requires ``model``. Defaults to the default variant of
                the model.
            weights (``PathType | dict[str, Any] | None``): A checkpoint
                path or URL, a loaded checkpoint, or a bare state
                dictionary. The constructor downloads a URL first, then
                loads the weights into the module. They take precedence
                over ``model.weights`` of the config, and that case logs
                a warning.
            allow_empty_dataset (bool): When ``True``, a `DummyLoader`
                replaces a loader that fails to build. The model then
                runs without a dataset, for example to export existing
                weights. When ``False``, the error propagates.
            dataset_metadata (DatasetMetadata | None): The dataset
                metadata to use. ``None`` reads the ``dataset_metadata``
                key of the ``weights`` checkpoint, and otherwise builds
                the metadata from the train loader. The metadata of a
                `DummyLoader` train loader wins over both sources when
                ``loader.params`` holds ``class_names``, ``n_classes``,
                or ``n_keypoints``. That case logs a warning.

        Raises:
            ValueError: When ``variant`` comes without ``model``. When
                ``cfg`` and ``model`` come together. When ``model``
                names no packaged model, carries a malformed
                version suffix, or ``variant`` names no variant of it.
                When ``cfg`` is ``None`` and the ``weights`` checkpoint
                has no ``config`` key.
            RuntimeError: When the download of ``weights`` fails.
            NotImplementedError: When ``trainer.use_weighted_sampler`` is
                set in the config.

        """
        if variant is not None and model is None:
            raise ValueError(
                "'variant' requires 'model' to be specified as well."
            )
        if model is not None:
            if cfg is not None:
                raise ValueError("'cfg' and 'model' are mutually exclusive.")
            resolved = resolve_predefined_config(model, variant)
            cfg = resolved.path
            if isinstance(opts, dict):
                model_opts = dict(
                    zip(
                        resolved.opts[::2],
                        resolved.opts[1::2],
                        strict=True,
                    )
                )
                opts = opts | model_opts
            elif resolved.opts:
                opts = [*(opts or []), *resolved.opts]

        restored_predefined_model: dict[str, Any] | None = None
        if weights is not None:
            weights, ckpt = self._normalize_weights_and_load_ckpt(weights)
            cfg, dataset_metadata, restored_predefined_model = (
                self._restore_cfg_and_metadata(ckpt, cfg, dataset_metadata)
            )

        if isinstance(cfg, Config):
            self.cfg = cfg
        else:
            self.cfg = Config.get_config(cfg, opts)

        self._allow_empty_dataset = allow_empty_dataset
        self._weights_provided_during_init = weights is not None
        self._weights_provided_in_config = self.cfg.model.weights is not None
        self._weights = weights or self.cfg.model.weights

        self.cfg_preprocessing = self.cfg.trainer.preprocessing

        self._init_tracker_and_trainer()
        self._build_loaders()
        self._build_pytorch_loaders()

        self._dataset_metadata = self._resolve_dataset_metadata(
            dataset_metadata
        )
        logger.info(f"Dataset metadata: {self._dataset_metadata}")
        self._config_file = self.run_save_dir / "training_config.yaml"
        self.cfg.save_data(self._config_file)

        self._input_shapes = self.loaders["train"].input_shapes

        self.lightning_module = LuxonisLightningModule(
            cfg=self.cfg,
            dataset_metadata=self._dataset_metadata,
            save_dir=self.run_save_dir,
            input_shapes=self._input_shapes,
            _core=self,
        )
        self.lightning_module._ckpt_predefined_model = (
            restored_predefined_model
        )

        if weights is not None:
            self._load_initial_weights(weights)
        self._exported_models: dict[str, Path] = {}

    @staticmethod
    @typechecked
    def _normalize_weights_and_load_ckpt(
        weights: PathType | dict[str, Any],
    ) -> tuple[PathType | dict[str, Any], dict[str, Any]]:
        if isinstance(weights, dict):
            if "state_dict" not in weights:
                weights = {"state_dict": weights}
            return weights, weights
        if isinstance(weights, PathType):
            downloaded = safe_download(weights)
            if downloaded is None:
                raise RuntimeError(
                    f"Failed to download weights from {weights}."
                )
            ckpt = torch.load(downloaded, map_location="cpu")  # nosemgrep
            return downloaded, ckpt
        raise ValueError(  # pragma: no cover
            f"Invalid type for weights: {type(weights)}. Expected str or dict."
        )

    @staticmethod
    def _restore_cfg_and_metadata(
        ckpt: dict[str, Any],
        cfg: PathType | Params | Config | None,
        dataset_metadata: DatasetMetadata | None,
    ) -> tuple[
        PathType | Params | Config,
        DatasetMetadata | None,
        dict[str, Any] | None,
    ]:
        restored_predefined_model = None
        if cfg is None:
            cfg = ckpt.get("config")
            if cfg is None:
                raise ValueError(
                    "Checkpoint does not contain the 'config' key. "
                    "Cannot restore `LuxonisModel` from checkpoint."
                )
            # The dumped config has no `predefined_model` block, so
            # provenance is inherited from the checkpoint — but only
            # when the config itself came from that checkpoint. A
            # user-supplied config warm-started from a checkpoint
            # must not claim the checkpoint's predefined model.
            restored_predefined_model = ckpt.get("predefined_model")
        if "dataset_metadata" in ckpt and dataset_metadata is None:
            try:
                dataset_metadata = DatasetMetadata(**ckpt["dataset_metadata"])
            except Exception as e:  # pragma: no cover
                logger.error(
                    "Failed to load dataset metadata from the checkpoint. "
                    f"Error: {e}"
                )
        return cfg, dataset_metadata, restored_predefined_model

    def _init_tracker_and_trainer(self) -> None:
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
        self._log_file = self.run_save_dir / "luxonis_train.log"
        self.error_message = None

        setup_logging(file=self._log_file, use_rich=self.cfg.rich_logging)

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

    def _build_loaders(self) -> None:
        self.loaders: dict[View, BaseLoaderTorch] = {}
        loader_name = self.cfg.loader.name
        Loader = LOADERS.get(loader_name)
        self._maybe_filter_task_names(Loader)

        for view in ("train", "val", "test"):
            if (
                view != "train"
                and issubclass(Loader, LuxonisLoaderTorch)
                and self.cfg.loader.params.get("dataset_dir") is not None
            ):
                self.cfg.loader.params["delete_existing"] = False

            self.loaders[view] = self._create_loader(view, Loader, loader_name)

        self._log_loader_sizes()

    def _maybe_filter_task_names(self, Loader: type) -> None:
        if not issubclass(Loader, LuxonisLoaderTorch):
            return
        model_tasks = {node.task_name for node in self.cfg.model.head_nodes}
        if model_tasks and None not in model_tasks:
            logger.info(
                f"Using {model_tasks} to filter task names from the dataset"
            )
            self.cfg.loader.params["filter_task_names"] = sorted(
                model_tasks  # type: ignore
            )

    def _create_loader(
        self, view: View, Loader: type, loader_name: str
    ) -> BaseLoaderTorch:
        view_name = {
            "train": self.cfg.loader.train_view,
            "val": self.cfg.loader.val_view,
            "test": self.cfg.loader.test_view,
        }[view]
        try:
            return Loader(
                view=view_name,
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
            if not self._allow_empty_dataset:
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
            return DummyLoader(
                cfg=self.cfg,
                view=view_name,
                image_source=self.cfg.loader.image_source,
                height=self.cfg_preprocessing.train_image_size.height,
                width=self.cfg_preprocessing.train_image_size.width,
                color_space=self.cfg_preprocessing.color_space,
                **self.cfg.loader.params,  # type: ignore
            )

    def _log_loader_sizes(self) -> None:
        for name, loader in self.loaders.items():
            logger.info(
                f"{name.capitalize()} loader - view: {loader.view}, size: {len(loader)}"
            )
            if len(loader) == 0:
                logger.warning(f"{name.capitalize()} loader is empty!")

    def _build_pytorch_loaders(self) -> None:
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
            loader, generator = self._resolve_eval_subset(view)
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
                generator=generator,
            )

    def _resolve_eval_subset(
        self, view: View
    ) -> tuple[
        BaseLoaderTorch | torch_data.Subset[LuxonisLoaderTorchOutput],
        torch.Generator | None,
    ]:
        n_val = self.cfg.trainer.n_validation_batches
        if n_val is None or view not in {"val", "test"}:
            return self.loaders[view], None

        generator = torch.Generator()
        generator.manual_seed(self.cfg.trainer.seed or 42)
        if n_val == -1:
            return self.loaders[view], generator

        n_samples = n_val * self.cfg.trainer.batch_size
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
        return torch_data.Subset(self.loaders[view], indices), generator

    def _resolve_dataset_metadata(
        self, dataset_metadata: DatasetMetadata | None
    ) -> DatasetMetadata:
        if dataset_metadata is None:
            return DatasetMetadata.from_loader(self.loaders["train"])
        if isinstance(self.loaders["train"], DummyLoader) and set(
            self.cfg.loader.params.keys()
        ).intersection({"class_names", "n_classes", "n_keypoints"}):
            logger.warning(
                "Dataset metadata from the checkpoint are "
                "overridden by extra loader parameters. "
                "The checkpoint metadata will not be used."
            )
            return DatasetMetadata.from_loader(self.loaders["train"])
        return dataset_metadata

    def _load_initial_weights(
        self, weights: PathType | dict[str, Any]
    ) -> None:
        if isinstance(weights, dict):
            if "state_dict" not in weights:
                weights = {"state_dict": weights}
            ckpt = weights
        else:
            ckpt = LuxonisFileSystem.download(str(weights), self.run_save_dir)
        if self.cfg.model.weights is not None:
            logger.warning(
                "Weights provided in the command line, but config weights are set. "
                "Ignoring weights provided in config."
            )
        self.lightning_module.load_checkpoint(ckpt)

    @property
    def train_loader(self) -> DataLoader:
        """The PyTorch data loader of the train view.

        It shuffles the samples, and it drops the last incomplete batch
        when ``trainer.skip_last_batch`` is set.

        """
        return self.pytorch_loaders["train"]

    @property
    def val_loader(self) -> DataLoader:
        """The PyTorch data loader of the validation view.

        It does not shuffle. When ``trainer.n_validation_batches`` is a
        positive number, it reads only that many batches, taken from the
        start of the view.

        """
        return self.pytorch_loaders["val"]

    @property
    def test_loader(self) -> DataLoader:
        """The PyTorch data loader of the test view.

        It does not shuffle. When ``trainer.n_validation_batches`` is a
        positive number, it reads only that many batches, taken from the
        start of the view.

        """
        return self.pytorch_loaders["test"]

    def save_checkpoint(
        self,
        path: PathType,
        weights_only: bool = False,
        storage_options: Any = None,
    ) -> Path:
        """Save a checkpoint of the model through the trainer.

        Besides the state dictionary, the checkpoint holds the config,
        the dataset metadata, the package version, and the execution
        order of the leaf modules. It also holds ``predefined_model``
        when the config or the loaded checkpoint names one. The
        constructor can rebuild a `LuxonisModel` from it alone.

        Args:
            path (``PathType``): The path of the checkpoint file.
            weights_only (bool): When ``True``, leave out the optimizer,
                the scheduler, and the callback states.
            storage_options (``Any``): Options passed to the
                ``CheckpointIO`` plugin of the trainer.

        Returns:
            ``Path``: ``path`` as a `pathlib.Path`.

        Raises:
            AttributeError: When no module is attached to the trainer
                yet. The trainer attaches the module when it first runs
                a fit, a test, or a prediction.

        """
        self.pl_trainer.save_checkpoint(
            path, weights_only=weights_only, storage_options=storage_options
        )
        return Path(path)

    def get_checkpoint(self, weights_only: bool = False) -> dict[str, Any]:
        """Return the checkpoint of the model as a dictionary.

        The method saves a temporary ``.ckpt`` file with
        `save_checkpoint`, loads it on the CPU, and deletes the file.

        Args:
            weights_only (bool): When ``True``, leave out the optimizer,
                the scheduler, and the callback states.

        Returns:
            ``dict[str, Any]``: The loaded checkpoint, with the
            ``state_dict`` key and the metadata `save_checkpoint`
            describes.

        Raises:
            AttributeError: When no module is attached to the trainer
                yet. The trainer attaches the module when it first runs
                a fit, a test, or a prediction.

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
        """Train the model.

        When ``trainer.matmul_precision`` is set, the method applies it
        first. It then resolves the weights and runs ``Trainer.fit``
        with the train and validation loaders. At the end, also after a
        failure, it uploads the log and the config to the run and
        finalizes the tracker.

        The weights come from ``weights``, else from the constructor,
        else from ``model.weights`` of the config:

        - With ``trainer.resume_training`` set, they go to
          ``Trainer.fit`` as ``ckpt_path``, so the optimizer, the
          scheduler, and the epoch count continue. Without any weights,
          the method logs a warning and the training starts from
          scratch.
        - Without ``trainer.resume_training``, the method loads only
          the model weights, and the training state starts fresh.

        Args:
            new_thread (bool): When ``True``, run the training in a
                daemon thread, store it in ``self.thread``, and return
                at once. The message of an exception raised in that
                thread goes to ``self.error_message``. The method
                replaces the global ``threading.excepthook`` to do so.
            weights (``PathType | None``): A checkpoint path or URL. It
                takes precedence over the weights of the constructor and
                of the config.

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
        """Export the model to ONNX, or save its checkpoint again.

        When weights are available, the method loads them into the
        module for the export and restores the previous weights
        afterwards. The output stem is ``<run_save_dir>/export/<name>``
        by default, where ``<name>`` is ``exporter.name`` or
        ``model.name`` of the config. Each output file adds its suffix
        to that stem. The method creates the output directory.

        Without ``ckpt_only``, the method:

        - exports the module to ``<stem>.onnx`` with the options of the
          ``exporter.onnx`` section;
        - simplifies the graph with ``onnxsim``, unless
          ``exporter.onnx.disable_onnx_simplification`` is set, and
          duplicates the shared initializers when
          ``exporter.onnx.unique_onnx_initializers`` is set;
        - uploads the ONNX file, and every earlier export of this
          instance, to the run when ``exporter.upload_to_run`` is set.
          It also uploads them to ``exporter.upload_url`` when that is
          set;
        - writes a ``modelconverter`` config to ``<stem>.yaml`` and
          uploads it the same way. The file holds the ONNX path, the
          mean and scale values, the color encoding, and the input and
          output shapes. A model with several inputs gets no YAML file;
          the method logs an error instead.

        The ONNX export leaves the module in training mode.

        The mean and scale values come from ``exporter.mean_values`` and
        ``exporter.scale_values``. Without them, they come from the
        ``mean`` and ``std`` of ``trainer.preprocessing.normalize``,
        multiplied by 255, when that section is active. Otherwise they
        are ``None``.

        Args:
            save_path (``PathType | None``): The directory of the output
                files. A path with a suffix names the output stem
                instead. ``None`` selects ``<run_save_dir>/export``.
            weights (``PathType | dict[str, Any] | None``): A checkpoint
                path or URL, a loaded checkpoint, or a bare state
                dictionary. ``None`` falls back to the weights of the
                constructor, then to ``model.weights`` of the config.
            ignore_missing_weights (bool): When ``True``, do not warn
                when no weights are available.
            ckpt_only (bool): When ``True``, only save ``<stem>.ckpt``
                through the trainer and return its path. Use it to
                refresh the metadata of a checkpoint, such as the config
                or the execution order of the nodes, without an ONNX
                export.

        Returns:
            ``Path``: The path of the ONNX file, or of the ``.ckpt`` file
            when ``ckpt_only`` is set.

        """
        weights = self.resolve_weights(weights)

        if not ignore_missing_weights and weights is None:
            logger.warning(
                "No model weights specified. Exporting model without weights."
            )
        export_path = self._resolve_export_path(save_path)
        export_path.parent.mkdir(parents=True, exist_ok=True)

        if ckpt_only:
            return self._export_ckpt_only(export_path, weights)

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
            self._upload_export_artifact(path)

        if len(self._input_shapes) > 1:
            logger.error(
                "Generating modelconverter config for a model "
                "with multiple inputs is not implemented yet."
            )
            return onnx_save_path

        modelconverter_config = self._build_modelconverter_config(
            onnx_save_path, scale_values, mean_values, color_space
        )

        yaml_path = export_path.with_suffix(".yaml")
        with open(yaml_path, "w") as f:
            yaml.safe_dump(
                modelconverter_config,
                f,
                sort_keys=False,
                default_flow_style=False,
            )
        self._upload_export_artifact(yaml_path, name=str(yaml_path))

        return onnx_save_path

    def _resolve_export_path(self, save_path: PathType | None) -> Path:
        model_name = self.cfg.exporter.name or self.cfg.model.name
        if save_path is None:
            return self.run_save_dir / "export" / model_name
        save_path = Path(save_path)
        if save_path.suffix:
            return save_path.with_suffix("")
        return save_path / model_name

    def _export_ckpt_only(
        self, export_path: Path, weights: PathType | dict[str, Any] | None
    ) -> Path:
        logger.info("Re-exporting the checkpoint file.")
        with replace_weights(self.lightning_module, weights):
            # Needs to be called to attach the model to the trainer
            self.pl_trainer.strategy._lightning_module = self.lightning_module
            self.pl_trainer.save_checkpoint(
                str(export_path.with_suffix(".ckpt")), weights_only=False
            )
            logger.info(
                f"Checkpoint saved to {export_path.with_suffix('.ckpt')}"
            )
        return export_path.with_suffix(".ckpt")

    def _upload_export_artifact(
        self, path: PathType, name: str | None = None
    ) -> None:
        if self.cfg.exporter.upload_to_run:
            self.tracker.upload_artifact(path, name=name, typ="export")
        if self.cfg.exporter.upload_url is not None:  # pragma: no cover
            LuxonisFileSystem.upload(path, self.cfg.exporter.upload_url)

    def _build_modelconverter_config(
        self,
        onnx_save_path: Path,
        scale_values: list[float] | None,
        mean_values: list[float] | None,
        color_space: str,
    ) -> dict[str, Any]:
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
        return {
            "input_model": onnx_save_path,
            "scale_values": scale_values,
            "mean_values": mean_values,
            "encoding": {"from": color_space, "to": "BGR"},
            "inputs": inputs,
            "outputs": outputs,
        }

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
        """Run the test loop on one dataset view.

        When weights are available, the method loads them into the
        module for the test and restores the previous weights
        afterwards. It runs ``Trainer.test`` with the PyTorch loader of
        ``view`` and returns the values logged in that epoch.

        Args:
            new_thread (bool): When ``True``, run the test in a daemon
                thread, store it in ``self.thread``, and return the
                thread at once.
            view (``Literal["train", "val", "test"]``): The dataset view
                to test on.
            weights (``PathType | dict[str, Any] | None``): A checkpoint
                path or URL, a loaded checkpoint, or a bare state
                dictionary. ``None`` falls back to the weights of the
                constructor, then to ``model.weights`` of the config.
            finalize_tracker (bool): When ``True``, upload the log and
                the config to the run and finalize the tracker once the
                test ends, also after a failure. Set it to ``False`` when
                the run continues with an export or an archive, and call
                `finalize_run` at the end.

        Returns:
            ``Mapping[str, float] | Thread``: The logged values of the
            test epoch when ``new_thread`` is ``False``. The keys are
            ``test/loss``, ``test/loss/<node>/<loss>``,
            ``test/loss/<node>/<loss>/<sub>`` when
            ``trainer.log_sub_losses`` is set, and
            ``test/metric/<node>/<name>`` for every scalar metric
            value. ``<name>`` is the metric identifier, or a sub-metric
            name when ``trainer.log_sub_metrics`` is set. A matrix value
            goes to the tracker instead. In these keys, ``<node>`` is
            the node name, prefixed with ``<task>-`` when the node has
            a task name. The prefix is ``test`` for every ``view``. The
            started thread when ``new_thread`` is ``True``.

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
        """Upload the run metadata and finalize the tracker.

        The method uploads ``luxonis_train.log`` and
        ``training_config.yaml`` of the run as artifacts, then closes
        the tracker with ``status``. It flushes and closes TensorBoard.
        MLFlow marks the run ``FINISHED`` for ``"success"`` or
        ``"finished"`` and ``FAILED`` otherwise. Weights and Biases
        gets the exit code ``0`` for ``"success"`` and ``1`` otherwise.
        Both steps run on rank zero only. `train` and `test` call this
        method themselves.

        Args:
            status (str): The final status of the run, ``"success"`` or
                ``"failed"``.

        """
        self._upload_run_metadata()
        self.tracker._finalize(status)

    def _upload_run_metadata(self) -> None:
        self.tracker.upload_artifact(self._log_file, typ="logs")
        self.tracker.upload_artifact(self._config_file, typ="config")

    def infer(
        self,
        view: Literal["train", "val", "test"] = "val",
        save_dir: PathType | None = None,
        source_path: PathType | None = None,
        weights: PathType | dict[str, Any] | None = None,
    ) -> None:
        """Run inference and show or save the visualizations.

        The method puts the module in eval mode. When weights are
        available, it loads them into the module for the call and
        restores the previous weights afterwards. The source of the
        images depends on ``source_path``:

        - ``None``: the PyTorch loader of ``view``. With
          ``trainer.overfit_batches`` set and ``view="train"``, the
          method reads only that many batches.
        - A video file (``.mp4``, ``.mov``, ``.avi``, ``.mkv``, or
          ``.webm``): every frame of the video.
        - Any other file: that file, read as an image.
        - A directory: every image file directly inside it, selected by
          extension (``.bmp``, ``.jpg``, ``.jpeg``, ``.png``, ``.tif``,
          ``.tiff``, ``.dng``, ``.webp``, ``.mpo``, or ``.pfm``).

        The method first loads an image file or a directory into a
        temporary dataset named ``infer_from_directory``. It replaces an
        existing local dataset of that name, and deletes the temporary
        one afterwards.

        Without ``save_dir``, each visualizer opens an OpenCV window
        named ``<node>/<visualizer>``. Press ``q`` or ``Esc`` to stop.
        With ``save_dir``, the method creates the directory and writes
        the renders to it:

        - ``<image stem>_<node>_<visualizer>.png`` for an image or a
          directory;
        - ``<node>_<visualizer>_<index>.png`` for a dataset view;
        - ``<node>_<visualizer>.mp4`` for a video.

        Args:
            view (``Literal["train", "val", "test"]``): The dataset view
                to read when ``source_path`` is ``None``.
            save_dir (``PathType | None``): The directory of the renders.
                ``None`` shows them on screen instead.
            source_path (``PathType | None``): An image file, a video
                file, or a directory of images. ``None`` reads the
                dataset.
            weights (``PathType | dict[str, Any] | None``): A checkpoint
                path or URL, a loaded checkpoint, or a bare state
                dictionary. ``None`` falls back to the weights of the
                constructor, then to ``model.weights`` of the config.

        Raises:
            ValueError: When ``source_path`` is neither a file nor a
                directory.

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
        """Annotate the images of a directory into a new dataset.

        The method puts the module in eval mode. When weights are
        available, it loads them into the module for the call and
        restores the previous weights afterwards. It reads every image
        file directly inside ``dir_path``, selected by extension as in
        `infer`. The images pass through a temporary dataset named
        ``infer_from_directory``. It replaces an existing local dataset
        of that name, and the method deletes it afterwards. Each head
        among the output nodes turns its outputs into records with
        `BaseHead.annotate`, and the records go into a `LuxonisDataset
        <luxonis_ml.data.datasets.LuxonisDataset>` named
        ``dataset_name``. The method skips a record with a bounding box
        outside the clipping range. A non-empty dataset gets the splits
        ``train``, ``val``, and ``test`` in the ratio 0.8, 0.1, and 0.1.
        An empty one logs a warning. The method prints the dataset info
        at the end.

        Args:
            dir_path (``PathType``): The directory that holds the images.
            dataset_name (str): The name of the dataset to create.
            weights (``PathType | dict[str, Any] | None``): A checkpoint
                path or URL, a loaded checkpoint, or a bare state
                dictionary. ``None`` falls back to the weights of the
                constructor, then to ``model.weights`` of the config.
            bucket_storage (``Literal["local", "gcs"]``): The storage
                backend of the dataset.
            delete_local (bool): Delete an existing local dataset of the
                same name first. With ``False``, the records go into the
                existing dataset.
            delete_remote (bool): Also delete the remote copy of an
                existing dataset of the same name.
            team_id (str | None): The team that owns the dataset. ``None``
                reads ``LUXONISML_TEAM_ID`` from the environment.

        Returns:
            ``LuxonisDataset``: The new dataset with the generated
            annotations.

        Raises:
            ValueError: When ``dir_path`` is not a directory.

        """
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
        """Run an Optuna study over the ``tuner.params`` of the config.

        The method creates a parent tracker for the study and an Optuna
        study named ``tuner.study_name``. The study minimizes
        ``val/loss`` when ``tuner.monitor`` is ``"loss"`` and maximizes
        the main metric otherwise. It prunes with a median pruner when
        ``tuner.use_pruner`` is set. When ``tuner.storage`` is active,
        it keeps the study in that SQLAlchemy database. A study of the
        same name continues when ``tuner.continue_existing_study`` is
        set. The study runs ``tuner.n_trials`` trials, or until
        ``tuner.timeout`` seconds pass.

        Each trial samples the parameters and builds a new config from
        them. It removes the callbacks that would upload, export,
        archive, or test the model of one trial: ``UploadCheckpoint``,
        ``ExportOnTrainEnd``, ``ArchiveOnTrainEnd``, and
        ``TestOnTrainEnd``. It then trains a new
        `LuxonisLightningModule` on the loaders of this instance. A
        child tracker with ``is_sweep`` set logs the trial under the run
        name of this instance. The trial writes its
        ``training_config.yaml`` and its checkpoints to the run
        directory of this instance. The trial returns the monitored
        value.

        At the end, the method logs the best parameters and writes the
        trials to ``tuner_study.csv`` in the run directory. It also logs
        the best parameters to the parent tracker. Weights and Biases
        allows one run per process, so the method creates the parent
        run at that point instead.

        Raises:
            ValueError: When ``tuner.params`` is empty. Also when
                ``tuner.monitor`` is ``"metric"`` and no main metric or
                no matching logging key exists.

        """
        import optuna

        cfg_tuner = self.cfg.tuner
        if cfg_tuner is None:
            raise ValueError(
                "You have to specify the `tuner` section in config."
            )

        all_augs = [a.name for a in self.cfg_preprocessing.augmentations]
        self._init_parent_tracker()

        logger.info("Starting tuning...")

        pruner = (
            optuna.pruners.MedianPruner()
            if cfg_tuner.use_pruner
            else optuna.pruners.NopPruner()
        )
        storage = self._build_optuna_storage(cfg_tuner)

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
            lambda trial: self._tune_objective(trial, all_augs),
            n_trials=cfg_tuner.n_trials,
            timeout=cfg_tuner.timeout,
        )
        logger.info(
            f"Best study parameters: {study.best_params}. Cost: {study.best_value}."
        )

        study_df = study.trials_dataframe()
        study_df.to_csv(self.run_save_dir / "tuner_study.csv", index=False)

        logger.info(
            f"Optuna study results saved to {self.run_save_dir / 'tuner_study.csv'}."
        )

        self._parent_tracker.log_hyperparams(study.best_params)

        self._finalize_wandb_tuning(study)

    def _tune_objective(
        self, trial: "optuna.trial.Trial", all_augs: list[str]
    ) -> float:
        import optuna
        from optuna.integration import PyTorchLightningPruningCallback

        from .utils.tune_utils import rename_params_for_logging

        assert self.cfg.tuner is not None

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

        cfg, curr_params = self._build_tuning_config(trial, all_augs)
        cfg.trainer.callbacks = self._filter_tuning_callbacks(cfg)

        renamed_params = rename_params_for_logging(
            curr_params, self.cfg.tuner.params
        )
        child_tracker.log_hyperparams(renamed_params)

        cfg.save_data(run_save_dir / "training_config.yaml")
        cfg.trainer.n_sanity_val_steps = 0
        lightning_module = LuxonisLightningModule(
            cfg=cfg,
            dataset_metadata=self._dataset_metadata,
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

        monitor = self._resolve_tuning_monitor(cfg)

        pruner_callback = PyTorchLightningPruningCallback(
            trial, monitor=monitor
        )
        callbacks.append(pruner_callback)
        callbacks.append(
            GracefulInterruptCallback(self.run_save_dir, self.tracker)
        )
        callbacks.append(FailOnNoTrainBatches())

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

    def _build_tuning_config(
        self, trial: "optuna.trial.Trial", all_augs: list[str]
    ) -> tuple[Config, Params]:
        from .utils.tune_utils import get_trial_params

        assert self.cfg.tuner is not None
        curr_params = get_trial_params(all_augs, self.cfg.tuner.params, trial)
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
        return cfg, curr_params

    @staticmethod
    def _filter_tuning_callbacks(cfg: Config) -> list[CallbackConfig]:
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
                    f"Callback '{cb.name}' is not supported for tuning and is removed from the callbacks list."
                )
            else:
                filtered_callbacks.append(cb)
        return filtered_callbacks

    def _resolve_tuning_monitor(self, cfg: Config) -> str:
        if cfg.tuner.monitor == "loss":
            return "val/loss"

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
        return monitor

    def _init_parent_tracker(self) -> None:
        rank = rank_zero_only.rank
        cfg_tracker = self.cfg.tracker
        tracker_params = get_tracker_init_params(cfg_tracker)
        # NOTE: wandb doesn't allow multiple concurrent runs, handle this separately
        tracker_params["is_wandb"] = False
        tracker_params["run_name"] = (
            tracker_params["run_name"] or self.tracker.run_name
        )
        self._parent_tracker = LuxonisTrackerPL(
            rank=rank,
            mlflow_tracking_uri=self.environ.MLFLOW_TRACKING_URI,
            is_sweep=False,
            **tracker_params,
        )
        if self._parent_tracker.is_mlflow:  # pragma: no cover
            # Experiment needs to be interacted with to create actual MLFlow run
            self._parent_tracker.experiment["mlflow"].active_run()

    @staticmethod
    def _build_optuna_storage(cfg_tuner: TunerConfig) -> "URL | None":
        from sqlalchemy import URL

        if not cfg_tuner.storage.active:
            return None
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
        return storage

    def _finalize_wandb_tuning(self, study: "optuna.study.Study") -> None:
        if self.cfg.tracker.is_wandb:  # pragma: no cover
            # If wandb used then init parent tracker separately at the end
            wandb_parent_tracker = LuxonisTrackerPL(
                rank=rank_zero_only.rank,
                _auto_finalize=True,
                **(
                    get_tracker_init_params(self.cfg.tracker)
                    | {"run_name": self._parent_tracker.run_name}
                ),
            )
            wandb_parent_tracker.log_hyperparams(study.best_params)

    def archive(
        self,
        path: PathType | None = None,
        weights: PathType | dict[str, Any] | None = None,
        save_dir: PathType | None = None,
    ) -> Path:
        """Generate an NN Archive from a model executable.

        When weights are available, the method loads them into the
        module for the call and restores the previous weights
        afterwards. Without ``path``, the method logs a warning and uses
        the ONNX file of the last `export` of this instance. When there
        is none, it runs `export` first. That export resolves its own
        weights from the constructor and the config, not from
        ``weights``.

        The archive is a ``.tar.xz`` file in ``save_dir``, or in
        ``<run_save_dir>/archive``. Its name is ``archiver.name`` or
        ``model.name`` of the config, plus the suffix of the executable,
        as in ``model.onnx.tar.xz``. It holds the executable and the
        ``<executable>.data`` external data file when one exists next
        to it. It also holds a ``config.json`` with the inputs, the
        outputs, and the head configs of the model. The method leaves
        out the heads with ``remove_on_export`` set. Each input carries
        the mean and scale of ``exporter.mean_values`` and
        ``exporter.scale_values``, or of ``trainer.preprocessing.normalize``
        multiplied by 255 when that section is active. Each input also
        carries the ``dai_type`` ``<color_space>888p``. The method
        uploads the archive to ``archiver.upload_url`` when that is set,
        and to the run when ``archiver.upload_to_run`` is set.

        Args:
            path (``PathType | None``): The model executable. It must be
                an ONNX file. ``None`` uses the last exported ONNX file,
                after an export when needed.
            weights (``PathType | dict[str, Any] | None``): A checkpoint
                path or URL, a loaded checkpoint, or a bare state
                dictionary. ``None`` falls back to the weights of the
                constructor, then to ``model.weights`` of the config.
            save_dir (``PathType | None``): The directory of the archive.
                ``None`` selects ``<run_save_dir>/archive``.

        Returns:
            ``Path``: The path of the archive.

        Raises:
            NotImplementedError: When ``path`` is not an ONNX file.

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
        """Export, archive, and convert the model for a device.

        The method runs `export` and `archive` with ``weights`` and
        ``save_dir``, then converts the result:

        - When ``exporter.blobconverter.active`` is set, it converts the
          ONNX file to a ``.blob`` for RVC2 with ``blobconverter``. The
          blob is ``FP16``, unless ``exporter.quantization_mode`` is
          ``FP32_STANDARD``. It reverses the input channels when
          ``exporter.reverse_input_channels`` is set. When that is
          ``None``, it reverses them when
          ``trainer.preprocessing.color_space`` is ``"RGB"``.
          ``blobconverter`` is deprecated, and a warning says so.
        - When ``exporter.hubai.active`` is set, it converts the archive
          for ``exporter.hubai.platform`` through the HubAI SDK. The SDK
          needs the ``HUBAI_API_KEY`` environment variable. The SDK
          names the variant on HubAI after the model and the train
          dataset. When
          ``exporter.hubai.delete_remote_model`` is set, the SDK call
          deletes the model or the variant it created on HubAI
          afterwards.

        The method skips a conversion whose package is not installed
        and logs it. It uploads the ``.blob`` like the exported files,
        and the HubAI archive like the archive.

        Args:
            weights (``PathType | dict[str, Any] | None``): A checkpoint
                path or URL, a loaded checkpoint, or a bare state
                dictionary. ``None`` falls back to the weights of the
                constructor, then to ``model.weights`` of the config.
            save_dir (``PathType | None``): The directory of every output
                file. ``None`` selects the run directory for the
                conversions, and its ``export`` and ``archive``
                subdirectories for the ONNX file and the archive.

        Returns:
            ``tuple[Path, dict[str, Path]]``: The path of the ONNX archive,
            and the conversion outputs keyed by ``"blob"`` and
            ``"hubai_archive"``. A key is present only when that
            conversion ran.

        Raises:
            RuntimeError: When the export produced no ONNX file.
            ValueError: When the HubAI conversion fails, for example
                without ``HUBAI_API_KEY``.

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
        reverse_input_channels = self._resolve_reverse_input_channels(
            color_space
        )

        convert_save_dir = (
            Path(save_dir) if save_dir else Path(self.run_save_dir)
        )

        conversion_artifacts: dict[str, Path] = {}
        blob_path = self._convert_blobconverter(
            onnx_path,
            scale_values,
            mean_values,
            reverse_input_channels,
            convert_save_dir,
        )
        if blob_path is not None:
            conversion_artifacts["blob"] = blob_path

        hubai_archive_path = self._convert_hubai(
            archive_path, convert_save_dir
        )
        if hubai_archive_path is not None:
            conversion_artifacts["hubai_archive"] = hubai_archive_path

        return archive_path, conversion_artifacts

    def _resolve_reverse_input_channels(self, color_space: str) -> bool:
        if self.cfg.exporter.reverse_input_channels is not None:
            return self.cfg.exporter.reverse_input_channels
        logger.info(
            "`exporter.reverse_input_channels` not specified. "
            "Using the `trainer.preprocessing.color_space` value "
            "to determine if the channels should be reversed. "
            f"`color_space` = '{color_space}' -> "
            f"`reverse_input_channels` = `{color_space == 'RGB'}`"
        )
        return color_space == "RGB"

    def _convert_blobconverter(
        self,
        onnx_path: Path,
        scale_values: list[float] | None,
        mean_values: list[float] | None,
        reverse_input_channels: bool,
        convert_save_dir: Path,
    ) -> Path | None:
        if not self.cfg.exporter.blobconverter.active:
            return None
        logger.warning(
            "blobconverter is deprecated and only supports RVC2 legacy conversion to `.blob`. "
            "Please consider using the HubAI SDK instead."
        )
        try:
            blob_path = Path(
                blobconverter_export(
                    self.cfg.exporter,
                    scale_values,
                    mean_values,
                    reverse_input_channels,
                    str(convert_save_dir),
                    str(onnx_path),
                )
            )
            self._exported_models["blob"] = blob_path
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
            return None
        else:
            return blob_path

    def _convert_hubai(
        self, archive_path: Path, convert_save_dir: Path
    ) -> Path | None:
        if not self.cfg.exporter.hubai.active:
            return None
        try:
            hubai_archive_path = Path(
                hubai_export(
                    cfg=self.cfg.exporter.hubai,
                    quantization_mode=self.cfg.exporter.quantization_mode,
                    archive_path=archive_path,
                    export_path=convert_save_dir,
                    model_name=self.cfg.model.name,
                    dataset_name=self._get_train_dataset_name(),
                )
            )
            self._exported_models["hubai_archive"] = hubai_archive_path
            if self.cfg.archiver.upload_to_run:
                self.tracker.upload_artifact(hubai_archive_path, typ="archive")
            if self.cfg.archiver.upload_url is not None:
                LuxonisFileSystem.upload(
                    hubai_archive_path, self.cfg.archiver.upload_url
                )
        except ImportError:
            logger.exception(
                "Unable to import `hubai_sdk`, skipping HubAI conversion."
            )
            return None
        except ValueError as e:
            raise ValueError(f"HubAI conversion failed: {e}") from e
        else:
            return hubai_archive_path

    def _get_train_dataset_name(self) -> str | None:
        loader = self.loaders["train"]
        if not isinstance(loader, LuxonisLoaderTorch):
            return None
        return loader.dataset.identifier

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
        """Quantize the model with AIMET, and export the result.

        The method needs the ``aimet`` extra. It writes every output to
        ``<run_save_dir>/aimet``. Each argument from ``epochs`` to
        ``scheduler`` that is ``None`` falls back to the matching field
        of ``exporter.aimet`` in the config.

        The steps are:

        - Build the model to quantize: a deep copy of the module, or
          the module itself when ``in_place`` is set. Reparameterize it,
          put it in eval mode, and load ``weights`` when given.
        - Test it on the validation loader, with inference mode off.
        - Run post-training quantization. Fold the batch norms and
          apply cross-layer equalization, AdaRound, and sequential MSE
          as configured. Then compute the encodings on the validation
          images, limited to the first
          ``exporter.aimet.max_calibration_images`` when that is set.
          Test the result.
        - Run quantization-aware training on the train loader for
          ``epochs`` epochs with ``optimizer`` and ``scheduler``. Then
          re-estimate and fold the batch norms as configured. Test the
          result.
        - Export the quantized model to ``<model.name>.onnx``, rename
          its outputs to the export names, and build an NN Archive from
          it in the same directory. The archive uploads as in
          `archive`.
        - Print a table with the results of the three tests.

        The three tests log their values to the tracker of the run.

        Args:
            weights (``PathType | None``): A checkpoint to load into the
                model before quantization. It does not fall back to the
                config weights, which the module already holds.
            epochs (int | None): The number of epochs of
                quantization-aware training.
            quant_scheme (``Literal["min_max", "tf", "tf_enhanced"] | None``):
                The rule that selects the quantization ranges.
            default_output_bw (int | None): The bit width of the
                activations.
            default_param_bw (int | None): The bit width of the
                parameters.
            config_file (str | None): The path of an AIMET config JSON
                file. When the fallback ``exporter.aimet.config`` is a
                dictionary, the method writes it to ``aimet_config.json``
                in the output directory first. Without either,
                ``batch_norm_reestimation`` selects the per-channel
                config of AIMET.
            default_data_type (``Literal["int", "float"] | None``): The
                data type of a quantized value.
            adaround (bool | None): Learn the rounding of the weights
                with AdaRound.
            adaround_iterations (int | None): The number of AdaRound
                iterations.
            adaround_reg_param (float | None): The AdaRound
                regularization parameter.
            adaround_beta_range (tuple[int, int] | None): The start and
                the end of the AdaRound beta annealing.
            adaround_warm_start (float | None): The share of the AdaRound
                iterations during which the rounding loss has no effect.
            fold_batch_norms (bool | None): Fold the batch norms into the
                preceding layers: before quantization when
                ``batch_norm_reestimation`` is off, and after
                quantization-aware training otherwise.
            cross_layer_equalization (bool | None): Balance the weight
                ranges across consecutive layers before quantization.
            batch_norm_reestimation (bool | None): Re-estimate the batch
                norm statistics after quantization-aware training.
            sequential_mse (bool | None): Optimize the quantization of
                each layer against the output of the float model.
            optimizer (``Optimizer | None``): The optimizer of
                quantization-aware training. ``None`` builds
                ``exporter.aimet.optimizer`` from the registry.
            scheduler (``LRScheduler | None``): The scheduler of
                quantization-aware training. ``None`` builds
                ``exporter.aimet.scheduler`` from the registry.
            in_place (bool): When ``True``, quantize ``lightning_module``
                itself, which saves memory but overwrites its weights
                and structure. The call also leaves it in export mode.
                When ``False``, quantize a deep copy.

        Returns:
            ``Path``: The output directory, ``<run_save_dir>/aimet``.

        Raises:
            ImportError: When ``aimet_torch`` is not installed.
            NotImplementedError: When the model has more than one input.

        """
        from .utils.aimet_utils import (
            check_aimet_available,
            get_ptq_calibration_loader,
            quantization_aware_training,
        )

        check_aimet_available()

        save_dir = self.run_save_dir / "aimet"
        save_dir.mkdir(parents=True, exist_ok=True)

        cfg = self.cfg.exporter.aimet
        aimet_config_file = self._prepare_aimet_config_file(
            config_file, cfg, save_dir
        )

        (
            adaround,
            fold_batch_norms,
            cross_layer_equalization,
            batch_norm_reestimation,
            sequential_mse,
        ) = self._resolve_aimet_overrides(
            cfg,
            adaround,
            fold_batch_norms,
            cross_layer_equalization,
            batch_norm_reestimation,
            sequential_mse,
        )

        model = self._build_quant_model(in_place, weights)

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

        dummy_inputs_dict, input_names, dummy_inputs = (
            self._build_dummy_inputs(model)
        )
        ptq_loader = get_ptq_calibration_loader(
            val_dataset=self.loaders["val"],
            collate_fn=self.loaders["val"].collate_fn,
            batch_size=self.cfg.trainer.batch_size,
            num_workers=self.cfg.trainer.n_workers,
            pin_memory=self.cfg.trainer.pin_memory,
            max_calibration_images=cfg.max_calibration_images,
        )

        sim = self._run_ptq(
            model,
            dummy_inputs,
            ptq_loader,
            save_dir,
            cfg,
            aimet_config_file,
            adaround,
            fold_batch_norms,
            cross_layer_equalization,
            batch_norm_reestimation,
            sequential_mse,
            quant_scheme,
            default_output_bw,
            default_param_bw,
            default_data_type,
            adaround_iterations,
            adaround_reg_param,
            adaround_beta_range,
            adaround_warm_start,
        )
        model = cast(LuxonisLightningModule, sim.model)

        model.eval()
        ptq_test = quant_eval_trainer.test(model, self.val_loader)[0]

        optimizer, scheduler = self._resolve_qat_optimizer_scheduler(
            optimizer, scheduler, sim, cfg
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

        self._export_quantized_onnx(
            sim, model, dummy_inputs, dummy_inputs_dict, input_names, save_dir
        )

        self._print_quant_results(model, pre_quant_test, ptq_test, qat_test)
        logger.info(f"AIMET artifacts saved in: {save_dir}")
        return save_dir

    @staticmethod
    def _prepare_aimet_config_file(
        config_file: str | None, cfg: AIMETConfig, save_dir: Path
    ) -> str | None:
        aimet_config_file = config_file or cfg.config
        if isinstance(aimet_config_file, dict):
            with open(save_dir / "aimet_config.json", "w") as f:
                json.dump(aimet_config_file, f, indent=4)
            aimet_config_file = str(save_dir / "aimet_config.json")
        return aimet_config_file

    @staticmethod
    def _resolve_aimet_overrides(
        cfg: AIMETConfig,
        adaround: bool | None,
        fold_batch_norms: bool | None,
        cross_layer_equalization: bool | None,
        batch_norm_reestimation: bool | None,
        sequential_mse: bool | None,
    ) -> tuple[bool, bool, bool, bool, bool]:
        def pick(override: bool | None, default: bool) -> bool:
            return default if override is None else override

        return (
            pick(adaround, cfg.adaround.active),
            pick(fold_batch_norms, cfg.fold_batch_norms),
            pick(cross_layer_equalization, cfg.cross_layer_equalization),
            pick(batch_norm_reestimation, cfg.batch_norm_reestimation),
            pick(sequential_mse, cfg.sequential_mse),
        )

    def _build_quant_model(
        self, in_place: bool, weights: PathType | None
    ) -> "LuxonisLightningModule":
        if not in_place:
            model = deepcopy(self.lightning_module)
        else:
            model = self.lightning_module

        model.reparameterize().eval()

        if weights is not None:
            model.load_checkpoint(weights)
        return model

    @staticmethod
    def _build_dummy_inputs(
        model: "LuxonisLightningModule",
    ) -> tuple[dict[str, Tensor], list[str], Tensor]:
        dummy_inputs_dict = {
            input_name: torch.randn([1, *shape]).to(model.device)
            for shapes in model.nodes.loader_input_shapes.values()
            for input_name, shape in shapes.items()
        }

        if len(dummy_inputs_dict) > 1:
            raise NotImplementedError(
                "Quantization is not yet supported for models "
                "with multiple inputs."
            )
        input_names = list(dummy_inputs_dict.keys())
        dummy_inputs = next(iter(dummy_inputs_dict.values()))
        return dummy_inputs_dict, input_names, dummy_inputs

    @staticmethod
    def _run_ptq(
        model: LuxonisLightningModule,
        dummy_inputs: Tensor,
        ptq_loader: DataLoader,
        save_dir: Path,
        cfg: AIMETConfig,
        aimet_config_file: str | None,
        adaround: bool,
        fold_batch_norms: bool,
        cross_layer_equalization: bool,
        batch_norm_reestimation: bool,
        sequential_mse: bool,
        quant_scheme: Literal["min_max", "tf", "tf_enhanced"] | None,
        default_output_bw: int | None,
        default_param_bw: int | None,
        default_data_type: Literal["int", "float"] | None,
        adaround_iterations: int | None,
        adaround_reg_param: float | None,
        adaround_beta_range: tuple[int, int] | None,
        adaround_warm_start: float | None,
    ) -> "QuantizationSimModel":
        from aimet_torch.common.defs import (  # pyright: ignore[reportMissingImports]
            QuantizationDataType,
            QuantScheme,
        )

        from .utils.aimet_utils import post_training_quantization

        return post_training_quantization(
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

    @staticmethod
    def _resolve_qat_optimizer_scheduler(
        optimizer: Optimizer | None,
        scheduler: LRScheduler | None,
        sim: "QuantizationSimModel",
        cfg: AIMETConfig,
    ) -> tuple[Optimizer, LRScheduler]:
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
        return optimizer, scheduler

    def _export_quantized_onnx(
        self,
        sim: "QuantizationSimModel",
        model: LuxonisLightningModule,
        dummy_inputs: Tensor,
        dummy_inputs_dict: dict[str, Tensor],
        input_names: list[str],
        save_dir: Path,
    ) -> None:
        model.set_export_mode(mode=True)
        output_names = model._get_output_onnx_names(
            deepcopy(dummy_inputs_dict)
        )

        onnx_path = (save_dir / self.cfg.model.name).with_suffix(".onnx")
        sim.onnx.export(
            dummy_inputs,
            onnx_path,
            input_names=input_names,
            output_names=output_names,
        )
        rename_onnx_outputs(onnx_path, output_names)
        self._archive(path=onnx_path, save_dir=save_dir)

    @staticmethod
    def _print_quant_results(
        model: LuxonisLightningModule,
        pre_quant_test: Mapping[str, float],
        ptq_test: Mapping[str, float],
        qat_test: Mapping[str, float],
    ) -> None:
        table = []
        for key, value in pre_quant_test.items():
            log_key = key.replace("test/metric/", "").replace("test/loss/", "")
            table.append((log_key, value, ptq_test[key], qat_test[key]))
        model.progress_bar.print_table(
            "Quantization results",
            table,
            ["Name", "Pre-Quant", "PTQ", "QAT"],
        )

    @property
    def environ(self) -> Environ:
        """The environment variables the config holds.

        This is the ``ENVIRON`` field of `Config`, an ``Environ``
        instance with fields such as ``MLFLOW_TRACKING_URI`` and the
        ``POSTGRES_*`` credentials.

        """
        return self.cfg.ENVIRON

    @rank_zero_only
    def get_min_loss_checkpoint_path(self) -> str | None:
        """Return the checkpoint path with the lowest validation loss.

        The method reads the ``best_model_path`` of the
        `lightning.pytorch.callbacks.ModelCheckpoint` callback that
        monitors ``val/loss``. The trainer attaches that callback when
        it first runs a fit, a test, or a prediction. It runs on rank
        zero only; every other rank gets ``None``.

        Returns:
            str | None: The checkpoint path, or ``None`` when no callback
            monitors ``val/loss``, as before the first run of the
            trainer. The path is an empty string while that callback
            has not saved a checkpoint yet.

        """
        for callback in self.pl_trainer.checkpoint_callbacks:
            if not isinstance(callback, ModelCheckpoint):
                continue
            if callback.monitor == "val/loss":
                return callback.best_model_path
        return None

    @rank_zero_only
    def get_best_metric_checkpoint_path(self) -> str | None:
        """Return the checkpoint path with the best validation metric.

        The method reads the ``best_model_path`` of the first
        ``ModelCheckpoint`` callback whose monitor contains
        ``val/metric/``. The trainer attaches that callback when it
        first runs a fit, a test, or a prediction. It runs on rank zero
        only; every other rank gets ``None``.

        Returns:
            str | None: The checkpoint path, or ``None`` when no callback
            monitors a validation metric, as before the first run of
            the trainer. The path is an empty string while that
            callback has not saved a checkpoint yet.

        """
        for callback in self.pl_trainer.checkpoint_callbacks:
            if not isinstance(callback, ModelCheckpoint):
                continue
            if callback.monitor and "val/metric/" in callback.monitor:
                return callback.best_model_path
        return None

    def get_mlflow_logging_keys(self) -> dict[str, list[str]]:
        """Return the keys a full run logs to MLFlow.

        The method delegates to
        `LuxonisLightningModule.get_mlflow_logging_keys`. The keys cover
        the losses, the metrics, and the metric artifacts such as
        confusion matrices. They also cover the visualizations of each
        evaluation epoch, the files the callbacks of the config produce,
        and the log and config files of the run.

        Returns:
            dict[str, list[str]]: The sorted metric keys under
            ``"metrics"`` and the sorted artifact names under
            ``"artifacts"``.

        """
        return self.lightning_module.get_mlflow_logging_keys()

    def resolve_weights(
        self, weights: PathType | dict[str, Any] | None
    ) -> PathType | dict[str, Any] | None:
        """Resolve the weights argument of a command to a local source.

        The precedence is ``weights``, then the weights of the
        constructor, then ``model.weights`` of the config. The method
        downloads a remote path to ``.cache/luxonis_train/<version>``.
        When ``weights`` is given, the config holds weights, and the
        constructor got none, the method logs a warning that it ignores
        the config weights.

        Args:
            weights (``PathType | dict[str, Any] | None``): A checkpoint
                path or URL, a loaded checkpoint, or a bare state
                dictionary. ``None`` selects the fallback.

        Returns:
            ``PathType | dict[str, Any] | None``: A local path, a
            checkpoint dictionary with a ``state_dict`` key, or ``None``
            when no weights exist or the download failed.

        """
        if isinstance(weights, dict):
            if "state_dict" not in weights:
                weights = {"state_dict": weights}
            return weights

        if weights is None:
            if isinstance(self._weights, dict):
                return self._weights
            return safe_download(self._weights)

        if (
            self._weights_provided_in_config
            and not self._weights_provided_during_init
        ):
            logger.warning(
                "Weights provided on the command line, but config weights are set. "
                "Ignoring weights provided in config."
            )
        return safe_download(weights)
