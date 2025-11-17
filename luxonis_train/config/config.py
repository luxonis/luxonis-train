import sys
from collections.abc import Mapping
from contextlib import suppress
from pathlib import Path
from typing import Annotated, Any, Literal, NamedTuple

from loguru import logger
from luxonis_ml.enums import DatasetType
from luxonis_ml.typing import (
    BaseModelExtraForbid,
    ConfigItem,
    Params,
    ParamValue,
    PathType,
    check_type,
)
from luxonis_ml.utils import (
    Environ,
    LuxonisConfig,
    LuxonisFileSystem,
    is_acyclic,
)
from pydantic import (
    AliasChoices,
    BeforeValidator,
    Field,
    PlainSerializer,
    SecretStr,
    SerializationInfo,
    field_validator,
    model_serializer,
    model_validator,
)
from pydantic.types import (
    FilePath,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveInt,
)
from pydantic_extra_types.semantic_version import SemanticVersion
from typing_extensions import Self, override

import luxonis_train as lxt
from luxonis_train.registry import MODELS, NODES, from_registry


class ImageSize(NamedTuple):
    height: int
    width: int


class AttachedModuleConfig(ConfigItem):
    alias: str | None = None

    @property
    def identifier(self) -> str:
        return self.alias or self.name


class LossModuleConfig(AttachedModuleConfig):
    weight: NonNegativeFloat = 1.0

    @model_validator(mode="after")
    def validate_weight(self) -> Self:
        if self.weight == 0:
            logger.warning(
                f"Loss '{self.name}' has weight set to 0. "
                "This loss will not contribute to the training."
            )
        return self


class MetricModuleConfig(AttachedModuleConfig):
    is_main_metric: bool = False


class FreezingConfig(BaseModelExtraForbid):
    active: bool = False
    unfreeze_after: NonNegativeInt | NonNegativeFloat | None = None
    lr_after_unfreeze: NonNegativeFloat | None = None


class NodeConfig(ConfigItem):
    alias: str | None = None
    inputs: list[str] = []  # From preceding nodes
    input_sources: list[str] = []  # From data loader
    freezing: FreezingConfig = Field(default_factory=FreezingConfig)
    remove_on_export: bool = False
    task_name: str | None = None
    metadata_task_override: str | dict[str, str] | None = None
    variant: (
        Annotated[str, BeforeValidator(str)]
        | Literal["default", "none"]
        | None
    ) = "default"
    losses: list[LossModuleConfig] = []
    metrics: list[MetricModuleConfig] = []
    visualizers: list[AttachedModuleConfig] = []

    @property
    def identifier(self) -> str:
        return self.alias or self.name


class PredefinedModelConfig(ConfigItem):
    variant: str | Literal["default", "none"] | None = "default"
    include_losses: bool = True
    include_metrics: bool = True
    include_visualizers: bool = True


class ModelConfig(BaseModelExtraForbid):
    name: str = "model"
    predefined_model: Annotated[
        PredefinedModelConfig | None, Field(exclude=True)
    ] = None
    weights: FilePath | None = None
    nodes: list[NodeConfig] = []
    outputs: list[str] = []

    @field_validator("nodes", mode="before")
    @classmethod
    def validate_nodes(cls, nodes: ParamValue) -> Any:
        logged_general_warning = False
        if not check_type(nodes, list[dict]):
            return nodes
        names = []
        last_body_index: int | None = None
        for i, node in enumerate(nodes):
            name = node.get("name")
            if name is None:
                raise ValueError(
                    f"Node {i} does not specify the `name` field."
                )
            if "Head" in name and last_body_index is None:
                last_body_index = i - 1
            name = node.get("alias") or name
            names.append(name)
            if i > 0 and "inputs" not in node and "input_sources" not in node:
                if last_body_index is not None:
                    prev_name = names[last_body_index]
                else:
                    prev_name = names[i - 1]

                if not logged_general_warning:
                    logger.warning(
                        f"Field `inputs` not specified for node '{name}'. "
                        "Assuming the model follows a linear multi-head topology "
                        "(backbone -> (neck?) -> head1, head2, ...). "
                        "If this is incorrect, please specify the `inputs` field explicitly."
                    )
                    logged_general_warning = True

                logger.warning(
                    f"Setting `inputs` of '{name}' to '{prev_name}'. "
                )
                node["inputs"] = [prev_name]
        return nodes

    @model_validator(mode="after")
    def validate_predefined_model(self) -> Self:
        if self.predefined_model is None:
            return self

        logger.info(f"Using predefined model: `{self.predefined_model.name}`")
        kwargs = dict(self.predefined_model.params or {})
        if not kwargs.get("variant"):
            kwargs["variant"] = self.predefined_model.variant
        model = from_registry(
            MODELS,
            self.predefined_model.name,
            **kwargs,
        )
        self.nodes += model.generate_nodes(
            include_losses=self.predefined_model.include_losses,
            include_metrics=self.predefined_model.include_metrics,
            include_visualizers=self.predefined_model.include_visualizers,
        )
        return self

    @model_validator(mode="after")
    def check_main_metric(self) -> Self:
        main_metric = None
        for node in self.nodes:
            for metric in node.metrics:
                if metric.is_main_metric:
                    if main_metric is not None:
                        raise ValueError(
                            f"Multiple main metrics specified: "
                            f"`{main_metric.identifier}` and "
                            f"`{metric.identifier}`. "
                            "Only one main metric can be specified."
                        )
                    main_metric = metric
                    logger.info(f"Main metric: `{metric.identifier}`")
        if main_metric is not None:
            return self

        logger.warning("No main metric specified.")
        all_metrics = [
            metric for node in self.nodes for metric in node.metrics
        ]
        if not all_metrics:
            logger.warning(
                "No metrics specified. "
                "This is likely unintended unless "
                "the configuration is not used for training."
            )
            return self

        all_metrics[0].is_main_metric = True
        logger.info(f"Setting '{all_metrics[0].identifier}' as main metric.")
        return self

    @model_validator(mode="after")
    def check_graph(self) -> Self:
        graph = {node.alias or node.name: node.inputs for node in self.nodes}
        if not is_acyclic(graph):
            raise ValueError("Model graph is not acyclic.")
        if not self.outputs:
            outputs: list[str] = []  # nodes which are not inputs to any nodes
            inputs = {
                node_name for node in self.nodes for node_name in node.inputs
            }
            for node in self.nodes:
                name = node.alias or node.name
                if name not in inputs:
                    outputs.append(name)
            self.outputs = outputs
        if self.nodes and not self.outputs:
            raise ValueError("No outputs specified.")
        return self

    @model_validator(mode="after")
    def check_for_invalid_characters(self) -> Self:
        for node in self.nodes:
            for modules in [
                node.losses,
                node.metrics,
                node.visualizers,
            ]:
                for module in [node, *modules]:
                    invalid_parts = []
                    if module.alias and "/" in module.alias:
                        invalid_parts.append(f"alias '{module.alias}'")
                    if module.name and "/" in module.name:
                        invalid_parts.append(f"name '{module.name}'")

                    if invalid_parts:
                        error_message = (
                            f"The {', '.join(invalid_parts)} contain a '/', which is not allowed. "
                            "Please rename to remove any '/' characters."
                        )
                        raise ValueError(error_message)

        return self

    @model_validator(mode="after")
    def check_unique_names(self) -> Self:
        for node in self.nodes:
            for modules in [
                node.losses,
                node.metrics,
                node.visualizers,
            ]:
                names: set[str] = set()
                node_index = 0
                for module in [node, *modules]:
                    module: AttachedModuleConfig | NodeConfig
                    name = module.alias or module.name
                    if name in names:
                        if module.alias is None:
                            if isinstance(module, NodeConfig):
                                module.alias = module.name
                            else:
                                module.alias = f"{name}_{node.alias}"

                        if module.alias in names:
                            new_alias = f"{module.alias}_{node_index}"
                            logger.warning(
                                f"Duplicate name: {module.alias}. Renaming to {new_alias}."
                            )
                            module.alias = new_alias
                            node_index += 1

                    names.add(name)
        return self

    @property
    def head_nodes(self) -> list[NodeConfig]:
        from luxonis_train.nodes import BaseHead

        return [
            node
            for node in self.nodes
            if issubclass(NODES._module_dict.get(node.name, object), BaseHead)
        ]


class TrackerConfig(BaseModelExtraForbid):
    project_name: str | None = None
    project_id: str | None = None
    run_name: str | None = None
    run_id: str | None = None
    save_directory: Path = Path("output")
    is_tensorboard: bool = True
    is_wandb: bool = False
    wandb_entity: str | None = None
    is_mlflow: bool = False


class LoaderConfig(ConfigItem):
    name: str = "LuxonisLoaderTorch"
    image_source: str = "image"
    train_view: list[str] = ["train"]
    val_view: list[str] = ["val"]
    test_view: list[str] = ["test"]

    @field_validator("train_view", "val_view", "test_view", mode="before")
    @classmethod
    def validate_view(cls, splits: ParamValue) -> list[Any]:
        if isinstance(splits, str):
            return [splits]
        if not isinstance(splits, list):
            raise TypeError(
                "Invalid value for `train_view`, `val_view`, "
                f"or `test_view`: {splits}. "
                "Expected a string or a list of strings."
            )
        return splits

    @model_validator(mode="after")
    def validate_params(self) -> Self:
        dataset_type = self.params.get("dataset_type")
        if dataset_type is None:
            return self
        if not isinstance(dataset_type, str):
            raise TypeError(
                f"Invalid value for `dataset_type`: {dataset_type}. "
                "Expected a string."
            )
        dataset_type = dataset_type.upper()

        if dataset_type not in DatasetType.__members__:
            raise ValueError(
                f"Dataset type '{dataset_type}' not supported."
                f"Supported types are: {', '.join(DatasetType.__members__)}."
            )
        self.params["dataset_type"] = dataset_type.lower()
        return self


class NormalizeAugmentationConfig(BaseModelExtraForbid):
    active: bool = True
    params: Params = {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    }


class AugmentationConfig(ConfigItem):
    active: bool = True


class PreprocessingConfig(BaseModelExtraForbid):
    train_image_size: Annotated[
        ImageSize, Field(min_length=2, max_length=2)
    ] = ImageSize(256, 256)
    keep_aspect_ratio: bool = True
    color_space: Literal["RGB", "BGR", "GRAY"] = "RGB"
    normalize: NormalizeAugmentationConfig = Field(
        default_factory=NormalizeAugmentationConfig
    )
    augmentations: list[AugmentationConfig] = []

    @model_validator(mode="after")
    def check_normalize(self) -> Self:
        norm = next(
            (aug for aug in self.augmentations if aug.name == "Normalize"),
            None,
        )
        if norm:
            if self.normalize.active:
                logger.warning(
                    "Normalize is being used in both trainer.preprocessing.augmentations "
                    "and trainer.preprocessing.normalize. "
                    "Parameters from trainer.preprocessing.augmentations list will override "
                    "those in trainer.preprocessing.normalize."
                )
            self.normalize.params = norm.params
            self.augmentations.remove(norm)

        if self.normalize.active:
            self.augmentations.append(
                AugmentationConfig(
                    name="Normalize", params=self.normalize.params
                )
            )
        return self

    @model_serializer
    def serialize_model(self, info: SerializationInfo) -> Params:
        data = {
            key: value
            for key, value in self.__dict__.items()
            if not key.startswith("_")
        }
        if "augmentations" in data and isinstance(data["augmentations"], list):
            data["augmentations"] = [
                aug
                for aug in data["augmentations"]
                if getattr(aug, "name", "") != "Normalize"
            ]

        return data

    def get_active_augmentations(self) -> list[ConfigItem]:
        """Returns list of augmentations that are active.

        @rtype: list[AugmentationConfig]
        @return: Filtered list of active augmentation configs
        """
        return [
            ConfigItem(name=aug.name, params=aug.params)
            for aug in self.augmentations
            if aug.active
        ]


class CallbackConfig(ConfigItem):
    active: bool = True


class TrainerConfig(BaseModelExtraForbid):
    preprocessing: PreprocessingConfig = Field(
        default_factory=PreprocessingConfig
    )

    precision: Literal["16-mixed", "32"] = "32"
    accelerator: Literal["auto", "cpu", "gpu", "tpu"] = "auto"
    devices: int | list[int] | str = "auto"
    strategy: Literal["auto", "ddp"] = "auto"
    n_sanity_val_steps: int = 2
    profiler: Literal["simple", "advanced"] | None = None
    matmul_precision: Literal["medium", "high", "highest"] | None = None

    seed: int | None = None
    n_validation_batches: PositiveInt | Literal[-1] | None = None
    deterministic: bool | Literal["warn"] | None = None
    smart_cfg_auto_populate: bool = True
    batch_size: PositiveInt = 32
    accumulate_grad_batches: PositiveInt | None = None
    gradient_clip_val: NonNegativeFloat | None = None
    gradient_clip_algorithm: Literal["norm", "value"] | None = None
    use_weighted_sampler: bool = False
    epochs: PositiveInt = 100
    resume_training: bool = False
    n_workers: NonNegativeInt = 4
    validation_interval: Literal[-1] | PositiveInt = 5
    n_log_images: NonNegativeInt = 4
    skip_last_batch: bool = True
    pin_memory: bool = True
    log_sub_losses: bool = True
    save_top_k: Literal[-1] | NonNegativeInt = 3

    callbacks: list[CallbackConfig] = []

    optimizer: ConfigItem = Field(
        default_factory=lambda: ConfigItem(name="Adam")
    )
    scheduler: ConfigItem = Field(
        default_factory=lambda: ConfigItem(name="ConstantLR")
    )

    training_strategy: ConfigItem | None = None

    @model_validator(mode="after")
    def validate_scheduler(self) -> Self:
        if self.scheduler.name == "CosineAnnealingLR":
            if "T_max" not in self.scheduler.params:
                self.scheduler.params["T_max"] = self.epochs
                logger.warning(
                    "`T_max` was not set for 'CosineAnnealingLR'"
                    "Automatically setting `T_max` to number of epochs."
                )
            elif self.scheduler.params["T_max"] != self.epochs:
                logger.warning(
                    "Parameter `T_max` of 'CosineAnnealingLR' is "
                    "not equal to the number of epochs. "
                    "Make sure this is intended."
                    f"`T_max`: {self.scheduler.params['T_max']}, "
                    f"Number of epochs: {self.epochs}"
                )

        return self

    @model_validator(mode="after")
    def validate_gradient_acc_scheduler(self) -> Self:
        """Keys in the GradientAccumulationSheduler.params.scheduling
        should be ints but yaml can sometime auto-convert them to
        strings.

        This converts them back to ints if possible.
        """
        for callback in self.callbacks:
            if callback.name != "GradientAccumulationScheduler":
                continue

            scheduling = callback.params.get("scheduling")
            if not isinstance(scheduling, Mapping):
                # Continue from Config verification standpoint but it might
                # fail due to GradientAccumulationScheduler param verification
                continue

            callback.params["scheduling"] = {
                int(k) if isinstance(k, str) and k.isdigit() else k: v
                for k, v in scheduling.items()
            }
        return self

    @model_validator(mode="after")
    def validate_deterministic(self) -> Self:
        if self.seed is not None and self.deterministic is None:
            logger.warning(
                "Setting `trainer.deterministic` to `True` because "
                "`trainer.seed` is set. This can cause certain "
                "layers to fail. In such cases, set "
                "`trainer.deterministic` to 'warn'."
            )
            self.deterministic = True
        return self

    @model_validator(mode="after")
    def check_n_workes_platform(self) -> Self:
        if (
            sys.platform == "win32" or sys.platform == "darwin"
        ) and self.n_workers != 0:
            self.n_workers = 0
            logger.warning(
                "Setting `n_workers` to 0 because of platform compatibility."
            )
        return self

    @model_validator(mode="after")
    def check_validation_interval(self) -> Self:
        if self.validation_interval > self.epochs:
            logger.warning(
                "Setting `validation_interval` same as `epochs`, "
                "otherwise no checkpoint would be generated."
            )
            self.validation_interval = self.epochs
        return self

    @model_validator(mode="after")
    def reorder_callbacks(self) -> Self:
        """Reorder callbacks so that EMA is the first callback, since it
        needs to be updated before other callbacks."""
        self.callbacks.sort(key=lambda v: 0 if v.name == "EMACallback" else 1)
        return self


class OnnxExportConfig(BaseModelExtraForbid):
    opset_version: PositiveInt = 16
    dynamic_axes: Params | None = None
    disable_onnx_simplification: bool = False


class BlobconverterExportConfig(BaseModelExtraForbid):
    active: bool = False
    shaves: int = 6
    version: Literal["2021.2", "2021.3", "2021.4", "2022.1", "2022.3_RVC3"] = (
        "2022.1"
    )


class ArchiveConfig(BaseModelExtraForbid):
    name: str | None = None
    upload_to_run: bool = True
    upload_url: str | None = None


class ExportConfig(ArchiveConfig):
    name: str | None = None
    input_shape: list[int] | None = None
    data_type: Literal["int8", "fp16", "fp32"] = "fp16"
    reverse_input_channels: bool | None = None
    scale_values: list[float] | None = None
    mean_values: list[float] | None = None
    onnx: OnnxExportConfig = Field(default_factory=OnnxExportConfig)
    blobconverter: BlobconverterExportConfig = Field(
        default_factory=BlobconverterExportConfig
    )

    @field_validator("scale_values", "mean_values", mode="before")
    @classmethod
    def check_values(cls, values: ParamValue) -> Any:
        if isinstance(values, float | int):
            return [values] * 3
        return values


class StorageConfig(BaseModelExtraForbid):
    active: bool = True
    backend: str = "sqlite"
    username: str | None = None
    password: SecretStr | None = None
    host: str | None = None
    port: PositiveInt | None = None
    database: str | None = None


class TunerConfig(BaseModelExtraForbid):
    study_name: str = "test-study"
    continue_existing_study: bool = True
    use_pruner: bool = True
    n_trials: PositiveInt | None = 15
    timeout: PositiveInt | None = None
    storage: StorageConfig = Field(default_factory=StorageConfig)
    params: dict[str, list[str | int | float | bool | list]] = {}
    monitor: Literal["metric", "loss"] = "loss"


class Config(LuxonisConfig):
    rich_logging: bool = True
    model: ModelConfig = Field(default_factory=ModelConfig)

    loader: LoaderConfig = Field(default_factory=LoaderConfig)
    tracker: TrackerConfig = Field(default_factory=TrackerConfig)
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    exporter: ExportConfig = Field(default_factory=ExportConfig)
    archiver: ArchiveConfig = Field(default_factory=ArchiveConfig)
    tuner: TunerConfig = Field(default_factory=TunerConfig)

    version: Annotated[
        SemanticVersion,
        Field(
            frozen=True,
            validation_alias=AliasChoices("version", "config_version"),
        ),
        PlainSerializer(str),
    ] = lxt.__semver__

    ENVIRON: Environ = Field(exclude=True, default_factory=Environ)

    @override
    def model_dump(
        self, exclude: set[str] | None = None, **kwargs
    ) -> dict[str, Any]:
        exclude = exclude or set()
        return super().model_dump(exclude=exclude | {"ENVIRON"}, **kwargs)

    @override
    def model_dump_json(
        self, exclude: set[str] | None = None, **kwargs
    ) -> str:
        exclude = exclude or set()
        return super().model_dump_json(exclude=exclude | {"ENVIRON"}, **kwargs)

    @model_validator(mode="before")
    @classmethod
    def check_environment(cls, data: Params) -> Params:
        if "ENVIRON" in data:
            logger.warning(
                "Specifying `ENVIRON` section in config file is not "
                "recommended due to security reasons. "
                "Please use environment variables or `.env` file instead."
            )
        return data

    @model_validator(mode="after")
    def check_tune_storage(self) -> Self:
        if self.tuner is None:
            return self
        stg = self.tuner.storage
        if stg.active:
            if stg.backend == "sqlite":
                if not stg.database:
                    stg.database = "study_local.db"
                    logger.warning(
                        "No database specified for SQLite storage. "
                        "Using default 'study_local.db'."
                    )
            elif stg.backend == "postgresql":
                stg.username = stg.username or self.ENVIRON.POSTGRES_USER
                stg.password = stg.password or self.ENVIRON.POSTGRES_PASSWORD
                stg.host = stg.host or self.ENVIRON.POSTGRES_HOST
                stg.port = stg.port or self.ENVIRON.POSTGRES_PORT
                stg.database = stg.database or self.ENVIRON.POSTGRES_DB

        return self

    @model_validator(mode="before")
    @classmethod
    def check_rich_logging(cls, data: Params) -> Params:
        use_rich = data.get("rich_logging", True)
        if not isinstance(use_rich, bool):
            raise TypeError(
                f"Invalid value for `rich_logging`: {use_rich}. "
                "Expected a boolean."
            )

        with suppress(ModuleNotFoundError):
            from luxonis_train.utils import setup_logging

            setup_logging(use_rich=use_rich)

        return data

    @classmethod
    def get_config(
        cls,
        cfg: PathType | Params | None = None,
        overrides: Params | list[str] | tuple[str, ...] | None = None,
    ) -> "Config":
        instance = super().get_config(cfg, overrides)
        if not isinstance(cfg, str):
            return instance.smart_auto_populate()
        fs = LuxonisFileSystem(cfg)
        if fs.is_mlflow:
            logger.info(
                "Setting `project_id` and `run_id` to config's MLFlow run"
            )
            instance.tracker.project_id = fs.experiment_id
            instance.tracker.run_id = fs.run_id

        if instance.trainer.smart_cfg_auto_populate:
            return instance.smart_auto_populate()

        return instance

    def smart_auto_populate(self) -> Self:
        """Automatically populates config fields based on rules, with
        warnings."""
        # Rule: Mosaic4 should have out_width and out_height
        # matching train_image_size if not provided
        for augmentation in self.trainer.preprocessing.augmentations:
            if augmentation.name == "Mosaic4" and (
                "out_width" not in augmentation.params
                or "out_height" not in augmentation.params
            ):
                train_size = self.trainer.preprocessing.train_image_size
                augmentation.params.update(
                    {"out_width": train_size[0], "out_height": train_size[1]}
                )
                logger.warning(
                    "`Mosaic4` augmentation detected. Automatically set `out_width` and `out_height` to match `train_image_size`."
                )

        # Rule: If all views are the same, set n_validation_batches
        if (
            self.loader.train_view
            == self.loader.val_view
            == self.loader.test_view
        ):
            if self.trainer.n_validation_batches is None:
                self.trainer.n_validation_batches = 10
                logger.warning(
                    "Train, validation, and test views are the same. "
                    "Automatically setting `n_validation_batches` to 10 "
                    "to prevent validation/testing on the full train set. "
                    "If this behavior is not desired, set "
                    "`smart_cfg_auto_populate` to `False`."
                )
            else:
                logger.warning(
                    "Train, validation, and test views are the same. "
                    "Make sure this is intended."
                )

        # Rule: Check if a predefined model is used and adjust
        # config accordingly to achieve best training results
        predefined_model_cfg = self.model.predefined_model
        if predefined_model_cfg is not None:
            logger.info(
                "Predefined model detected. "
                "Adjusting  parameters for best training results. "
                "If this behavior is not desired, set "
                "`smart_cfg_auto_populate` to `False`."
            )
            model_name = predefined_model_cfg.name
            accumulate_grad_batches = int(64 / self.trainer.batch_size)
            logger.info(
                f"Setting 'accumulate_grad_batches' to "
                f"{accumulate_grad_batches} "
                f"(trainer.batch_size={self.trainer.batch_size})",
                accumulate_grad_batches,
                self.trainer.batch_size,
            )
            loss_params = predefined_model_cfg.params.get("loss_params", {})
            if not isinstance(loss_params, dict):
                raise ValueError(
                    f"Invalid value for loss_params: {loss_params}. "
                    "Expected a dictionary."
                )
            gradient_accumulation_schedule = None
            if model_name == "InstanceSegmentationModel":
                loss_params.update(
                    {
                        "bbox_loss_weight": 7.5 * accumulate_grad_batches,
                        "class_loss_weight": 0.5 * accumulate_grad_batches,
                        "dfl_loss_weight": 1.5 * accumulate_grad_batches,
                    }
                )
                gradient_accumulation_schedule = {
                    0: 1,
                    1: (1 + accumulate_grad_batches) // 2,
                    2: accumulate_grad_batches,
                }
                logger.info(
                    f"InstanceSegmentationModel: Updated loss_params: {loss_params}"
                )
                logger.info(
                    f"InstanceSegmentationModel: Set gradient "
                    f"accumulation schedule to: {gradient_accumulation_schedule}"
                )
            elif model_name == "KeypointDetectionModel":
                loss_params.update(
                    {
                        "iou_loss_weight": 7.5 * accumulate_grad_batches,
                        "class_loss_weight": 0.5 * accumulate_grad_batches,
                        "regr_kpts_loss_weight": 12 * accumulate_grad_batches,
                        "vis_kpts_loss_weight": 1 * accumulate_grad_batches,
                    }
                )
                gradient_accumulation_schedule = {
                    0: 1,
                    1: (1 + accumulate_grad_batches) // 2,
                    2: accumulate_grad_batches,
                }
                logger.info(
                    f"KeypointDetectionModel: Updated loss_params: {loss_params}"
                )
                logger.info(
                    f"KeypointDetectionModel: Set gradient accumulation "
                    f"schedule to: {gradient_accumulation_schedule}"
                )
            elif model_name == "DetectionModel":
                loss_params.update(
                    {
                        "iou_loss_weight": 2.5 * accumulate_grad_batches,
                        "class_loss_weight": 1 * accumulate_grad_batches,
                    }
                )
                logger.info(
                    f"DetectionModel: Updated loss_params: {loss_params}"
                )
            predefined_model_cfg.params["loss_params"] = loss_params
            if gradient_accumulation_schedule:
                for callback in self.trainer.callbacks:
                    if callback.name == "GradientAccumulationScheduler":
                        callback.params["scheduling"] = (  # type: ignore
                            gradient_accumulation_schedule
                        )
                        logger.info(
                            f"GradientAccumulationScheduler callback "
                            f"updated with scheduling: {gradient_accumulation_schedule}"
                        )
                        break

        default_callbacks = [
            "UploadCheckpoint",
            "TestOnTrainEnd",
            "ExportOnTrainEnd",
            "ArchiveOnTrainEnd",
        ]

        for cb_name in default_callbacks:
            if not any(cb.name == cb_name for cb in self.trainer.callbacks):
                self.trainer.callbacks.append(CallbackConfig(name=cb_name))
                logger.info(f"Added {cb_name} callback.")

        return self
