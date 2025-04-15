import sys
from pathlib import Path
from typing import Annotated, Any, Literal, NamedTuple

from loguru import logger
from luxonis_ml.enums import DatasetType
from luxonis_ml.typing import ConfigItem, Params, ParamValue, check_type
from luxonis_ml.utils import (
    BaseModelExtraForbid,
    Environ,
    LuxonisConfig,
    LuxonisFileSystem,
    is_acyclic,
)
from pydantic import Field, field_validator, model_validator
from pydantic.types import (
    FilePath,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveInt,
)
from typing_extensions import Self

from luxonis_train.registry import MODELS


class ImageSize(NamedTuple):
    height: int
    width: int


class AttachedModuleConfig(ConfigItem):
    attached_to: str
    alias: str | None = None


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


class NodeConfig(ConfigItem):
    alias: str | None = None
    inputs: list[str] = []  # From preceding nodes
    input_sources: list[str] = []  # From data loader
    freezing: FreezingConfig = Field(default_factory=FreezingConfig)
    remove_on_export: bool = False
    task_name: str = ""
    metadata_task_override: str | dict[str, str] | None = None


class PredefinedModelConfig(ConfigItem):
    include_nodes: bool = True
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
    losses: list[LossModuleConfig] = []
    metrics: list[MetricModuleConfig] = []
    visualizers: list[AttachedModuleConfig] = []
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
    def check_predefined_model(self) -> Self:
        if self.predefined_model is not None:
            logger.info(
                f"Using predefined model: `{self.predefined_model.name}`"
            )
            model = MODELS.get(self.predefined_model.name)(
                **self.predefined_model.params
            )
            nodes, losses, metrics, visualizers = model.generate_model(
                include_nodes=self.predefined_model.include_nodes,
                include_losses=self.predefined_model.include_losses,
                include_metrics=self.predefined_model.include_metrics,
                include_visualizers=self.predefined_model.include_visualizers,
            )
            self.nodes += nodes
            self.losses += losses
            self.metrics += metrics
            self.visualizers += visualizers

        return self

    @model_validator(mode="after")
    def check_main_metric(self) -> Self:
        for metric in self.metrics:
            if metric.is_main_metric:
                logger.info(f"Main metric: `{metric.name}`")
                return self

        logger.warning("No main metric specified.")
        if self.metrics:
            for metric in self.metrics:
                if "matrix" not in metric.name.lower():
                    metric.is_main_metric = True
                    name = metric.alias or metric.name
                    logger.info(f"Setting '{name}' as main metric.")
                    return self
            raise ValueError(
                "No valid main metric can be set as all "
                "metrics contain 'matrix' in their names."
            )
        logger.warning(
            "[Ignore if using predefined model] "
            "No metrics specified. "
            "This is likely unintended unless "
            "the configuration is not used for training."
        )
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
        for modules in [
            self.nodes,
            self.losses,
            self.metrics,
            self.visualizers,
        ]:
            for module in modules:
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
        for modules in [
            self.nodes,
            self.losses,
            self.metrics,
            self.visualizers,
        ]:
            names: set[str] = set()
            node_index = 0
            for module in modules:
                module: AttachedModuleConfig | NodeConfig
                name = module.alias or module.name
                if name in names:
                    if module.alias is None:
                        if isinstance(module, NodeConfig):
                            module.alias = module.name
                        else:
                            module.alias = f"{name}_{module.attached_to}"

                    if module.alias in names:
                        new_alias = f"{module.alias}_{node_index}"
                        logger.warning(
                            f"Duplicate name: {module.alias}. Renaming to {new_alias}."
                        )
                        module.alias = new_alias
                        node_index += 1

                names.add(name)
        return self

    @model_validator(mode="before")
    @classmethod
    def check_attached_modules(cls, data: Params) -> Params:
        if "nodes" not in data:
            return data
        for section in ["losses", "metrics", "visualizers"]:
            if section not in data:
                data[section] = []
            else:
                logger.warning(
                    f"Field `model.{section}` is deprecated. "
                    f"Please specify `{section}` under "
                    "the node they are attached to."
                )
            if not check_type(data["nodes"], list[dict[str, Any]]):
                raise ValueError(
                    "Invalid value for `model.nodes`. "
                    "Expected a list of dictionaries."
                )
            for node in data["nodes"]:
                if section in node:
                    cfg = node.pop(section)
                    if not isinstance(cfg, list):
                        cfg = [cfg]
                    for c in cfg:
                        c["attached_to"] = node.get("alias", node.get("name"))
                    section_data = data[section]
                    if not check_type(section_data, list[dict]):
                        raise ValueError(
                            f"Invalid value for `model.{section}`. "
                            "Expected a list of dictionaries."
                        )
                    section_data.extend(cfg)
        return data


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
        self.params["dataset_type"] = DatasetType(dataset_type.lower())
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
    color_space: Literal["RGB", "BGR"] = "RGB"
    normalize: NormalizeAugmentationConfig = Field(
        default_factory=NormalizeAugmentationConfig
    )
    augmentations: list[AugmentationConfig] = []

    @model_validator(mode="before")
    @classmethod
    def validate_train_rgb(cls, data: Params) -> Params:
        if "train_rgb" in data:
            logger.warning(
                "Field `train_rgb` is deprecated. Use `color_space` instead."
            )
            data["color_space"] = "RGB" if data.pop("train_rgb") else "BGR"
        return data

    @model_validator(mode="after")
    def check_normalize(self) -> Self:
        if self.normalize.active:
            self.augmentations.append(
                AugmentationConfig(
                    name="Normalize", params=self.normalize.params
                )
            )
        return self

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
    use_rich_progress_bar: bool = True

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
    opset_version: PositiveInt = 12
    dynamic_axes: Params | None = None


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
    output_names: list[str] | None = None
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
    storage_type: Literal["local", "remote"] = "local"


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
    model: ModelConfig = Field(default_factory=ModelConfig)

    loader: LoaderConfig = Field(default_factory=LoaderConfig)
    tracker: TrackerConfig = Field(default_factory=TrackerConfig)
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    exporter: ExportConfig = Field(default_factory=ExportConfig)
    archiver: ArchiveConfig = Field(default_factory=ArchiveConfig)
    tuner: TunerConfig | None = None

    ENVIRON: Environ = Field(exclude=True, default_factory=Environ)

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

    @classmethod
    def get_config(
        cls,
        cfg: str | Params | None = None,
        overrides: Params | list[str] | tuple[str, ...] | None = None,
    ) -> "Config":
        instance = super().get_config(cfg, overrides)
        if not isinstance(cfg, str):
            cls.smart_auto_populate(instance)
            return instance
        fs = LuxonisFileSystem(cfg)
        if fs.is_mlflow:
            logger.info(
                "Setting `project_id` and `run_id` to config's MLFlow run"
            )
            instance.tracker.project_id = fs.experiment_id
            instance.tracker.run_id = fs.run_id

        if instance.trainer.smart_cfg_auto_populate:
            cls.smart_auto_populate(instance)

        return instance

    @classmethod
    def smart_auto_populate(cls, instance: "Config") -> None:
        """Automatically populates config fields based on rules, with
        warnings."""

        # Rule: Mosaic4 should have out_width and out_height matching train_image_size if not provided
        for augmentation in instance.trainer.preprocessing.augmentations:
            if augmentation.name == "Mosaic4" and (
                "out_width" not in augmentation.params
                or "out_height" not in augmentation.params
            ):
                train_size = instance.trainer.preprocessing.train_image_size
                augmentation.params.update(
                    {"out_width": train_size[0], "out_height": train_size[1]}
                )
                logger.warning(
                    "`Mosaic4` augmentation detected. Automatically set `out_width` and `out_height` to match `train_image_size`."
                )

        # Rule: If train, val, and test views are the same, set n_validation_batches
        if (
            instance.loader.train_view
            == instance.loader.val_view
            == instance.loader.test_view
        ):
            if instance.trainer.n_validation_batches is None:
                instance.trainer.n_validation_batches = 10
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

        # Rule: Check if a predefined model is set and adjust config accordingly to achieve best training results
        predefined_model_cfg = instance.model.predefined_model
        if predefined_model_cfg is not None:
            logger.info(
                "Predefined model detected. "
                "Adjusting  parameters for best training results."
                "If this behavior is not desired, set "
                "`smart_cfg_auto_populate` to `False`."
            )
            model_name = predefined_model_cfg.name
            accumulate_grad_batches = int(64 / instance.trainer.batch_size)
            logger.info(
                f"Setting 'accumulate_grad_batches' to "
                f"{accumulate_grad_batches} "
                f"(trainer.batch_size={instance.trainer.batch_size})",
                accumulate_grad_batches,
                instance.trainer.batch_size,
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
                for callback in instance.trainer.callbacks:
                    if callback.name == "GradientAccumulationScheduler":
                        callback.params["scheduling"] = (  # type: ignore
                            gradient_accumulation_schedule
                        )
                        logger.info(
                            f"GradientAccumulationScheduler callback "
                            f"updated with scheduling: {gradient_accumulation_schedule}"
                        )
                        break
