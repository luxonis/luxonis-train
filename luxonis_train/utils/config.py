import logging
import sys
from enum import Enum
from typing import Annotated, Any, Literal

from luxonis_ml.data import BucketStorage, BucketType
from luxonis_ml.utils import Environ, LuxonisConfig, LuxonisFileSystem, setup_logging
from pydantic import BaseModel, Field, field_serializer, model_validator

from luxonis_train.utils.general import is_acyclic
from luxonis_train.utils.registry import MODELS

logger = logging.getLogger(__name__)


class AttachedModuleConfig(BaseModel):
    name: str
    attached_to: str
    override_name: str | None = None
    params: dict[str, Any] = {}


class LossModuleConfig(AttachedModuleConfig):
    weight: float = 1.0


class MetricModuleConfig(AttachedModuleConfig):
    is_main_metric: bool = False


class ModelNodeConfig(BaseModel):
    name: str
    override_name: str | None = None
    inputs: list[str] = []
    params: dict[str, Any] = {}
    frozen: bool = False


class PredefinedModelConfig(BaseModel):
    name: str
    params: dict[str, Any] = {}
    include_nodes: bool = True
    include_losses: bool = True
    include_metrics: bool = True
    include_visualizers: bool = True


class ModelConfig(BaseModel):
    name: str
    predefined_model: PredefinedModelConfig | None = None
    weights: str | None = None
    nodes: list[ModelNodeConfig] = []
    losses: list[LossModuleConfig] = []
    metrics: list[MetricModuleConfig] = []
    visualizers: list[AttachedModuleConfig] = []
    outputs: list[str] = []

    @model_validator(mode="after")
    def check_predefined_model(self):
        if self.predefined_model:
            logger.info(f"Using predefined model: `{self.predefined_model.name}`")
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
    def check_graph(self):
        graph = {node.override_name or node.name: node.inputs for node in self.nodes}
        if not is_acyclic(graph):
            raise ValueError("Model graph is not acyclic.")
        if not self.outputs:
            outputs: list[str] = []  # nodes which are not inputs to any nodes
            inputs = set(node_name for node in self.nodes for node_name in node.inputs)
            for node in self.nodes:
                name = node.override_name or node.name
                if name not in inputs:
                    outputs.append(name)
            self.outputs = outputs
        if self.nodes and not self.outputs:
            raise ValueError("No outputs specified.")
        return self

    model_config = {
        "json_schema_extra": {
            "if": {"properties": {"predefined_model": {"type": "null"}}},
            "then": {"properties": {"nodes": {"type": "array"}}},
        }
    }


class TrackerConfig(BaseModel):
    project_name: str | None = None
    project_id: str | None = None
    run_name: str | None = None
    run_id: str | None = None
    save_directory: str = "output"
    is_tensorboard: bool = True
    is_wandb: bool = False
    wandb_entity: str | None = None
    is_mlflow: bool = False


class DatasetConfig(BaseModel):
    dataset_name: str | None = None
    dataset_id: str | None = None
    team_name: str | None = None
    team_id: str | None = None
    bucket_type: BucketType = BucketType.INTERNAL
    bucket_storage: BucketStorage = BucketStorage.LOCAL
    json_mode: bool = False
    train_view: str = "train"
    val_view: str = "val"
    test_view: str = "test"

    @field_serializer("bucket_storage", "bucket_type")
    def get_enum_value(self, v: Enum, _) -> str:
        return str(v.value)

    model_config = {
        "json_schema_extra": {
            "anyOf": [
                {
                    "allOf": [
                        {"required": ["dataset_name"]},
                        {"properties": {"dataset_name": {"type": "string"}}},
                    ]
                },
                {
                    "allOf": [
                        {"required": ["dataset_id"]},
                        {"properties": {"dataset_id": {"type": "string"}}},
                    ]
                },
            ]
        },
    }


class NormalizeAugmentationConfig(BaseModel):
    active: bool = True
    params: dict[str, Any] = {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    }


class AugmentationConfig(BaseModel):
    name: str
    params: dict[str, Any] = {}


class PreprocessingConfig(BaseModel):
    train_image_size: Annotated[
        list[int], Field(default=[256, 256], min_length=2, max_length=2)
    ] = [256, 256]
    keep_aspect_ratio: bool = True
    train_rgb: bool = True
    normalize: NormalizeAugmentationConfig = NormalizeAugmentationConfig()
    augmentations: list[AugmentationConfig] = []

    @model_validator(mode="after")
    def check_normalize(self):
        if self.normalize.active:
            self.augmentations.append(
                AugmentationConfig(name="Normalize", params=self.normalize.params)
            )
        return self


class CallbackConfig(BaseModel):
    name: str
    active: bool = True
    params: dict[str, Any] = {}


class OptimizerConfig(BaseModel):
    name: str = "Adam"
    params: dict[str, Any] = {}


class SchedulerConfig(BaseModel):
    name: str = "ConstantLR"
    params: dict[str, Any] = {}


class TrainerConfig(BaseModel):
    preprocessing: PreprocessingConfig = PreprocessingConfig()

    accelerator: Literal["auto", "cpu", "gpu"] = "auto"
    devices: int | list[int] | str = "auto"
    strategy: Literal["auto", "ddp"] = "auto"
    num_sanity_val_steps: int = 2
    profiler: Literal["simple", "advanced"] | None = None
    verbose: bool = True

    batch_size: int = 32
    accumulate_grad_batches: int = 1
    use_weighted_sampler: bool = False
    epochs: int = 100
    num_workers: int = 2
    train_metrics_interval: int = -1
    validation_interval: int = 1
    num_log_images: int = 4
    skip_last_batch: bool = True
    log_sub_losses: bool = True
    save_top_k: int = 3

    callbacks: list[CallbackConfig] = []

    optimizer: OptimizerConfig = OptimizerConfig()
    scheduler: SchedulerConfig = SchedulerConfig()

    @model_validator(mode="after")
    def check_num_workes_platform(self):
        if (
            sys.platform == "win32" or sys.platform == "darwin"
        ) and self.num_workers != 0:
            self.num_workers = 0
            logger.warning(
                "Setting `num_workers` to 0 because of platform compatibility."
            )
        return self


class OnnxExportConfig(BaseModel):
    opset_version: int = 12
    dynamic_axes: dict[str, Any] | None = None


class BlobconverterExportConfig(BaseModel):
    active: bool = False
    shaves: int = 6


class ExportConfig(BaseModel):
    export_save_directory: str = "output_export"
    input_shape: list[int] | None = None
    export_model_name: str = "model"
    data_type: Literal["INT8", "FP16", "FP32"] = "FP16"
    reverse_input_channels: bool = True
    scale_values: list[float] | None = None
    mean_values: list[float] | None = None
    onnx: OnnxExportConfig = OnnxExportConfig()
    blobconverter: BlobconverterExportConfig = BlobconverterExportConfig()
    upload_url: str | None = None

    @model_validator(mode="after")
    def check_values(self):
        def pad_values(values: float | list[float] | None):
            if values is None:
                return None
            if isinstance(values, float):
                return [values] * 3

        self.scale_values = pad_values(self.scale_values)
        self.mean_values = pad_values(self.mean_values)
        return self


class StorageConfig(BaseModel):
    active: bool = True
    storage_type: Literal["local", "remote"] = "local"


class TunerConfig(BaseModel):
    study_name: str = "test-study"
    use_pruner: bool = True
    n_trials: int | None = 15
    timeout: int | None = None
    storage: StorageConfig = StorageConfig()
    params: Annotated[
        dict[str, list[str | int | float | bool]], Field(default={}, min_length=1)
    ] = {}

    model_config = {"json_schema_extra": {"required": ["params"]}}


class Config(LuxonisConfig):
    use_rich_text: bool = True
    model: ModelConfig
    dataset: DatasetConfig = DatasetConfig()
    tracker: TrackerConfig = TrackerConfig()
    trainer: TrainerConfig = TrainerConfig()
    exporter: ExportConfig = ExportConfig()
    tuner: TunerConfig = TunerConfig()
    ENVIRON: Environ = Field(Environ(), exclude=True)

    @model_validator(mode="before")
    @classmethod
    def check_tuner_init(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if data.get("tuner") and not data.get("tuner", {}).get("params"):
                del data["tuner"]
                logger.warning(
                    "`tuner` block specified but no `tuner.params`. If trying to tune values you have to specify at least one parameter"
                )
        return data

    @model_validator(mode="before")
    @classmethod
    def check_environment(cls, data: Any) -> Any:
        if "ENVIRON" in data:
            logger.warning(
                "Specifying `ENVIRON` section in config file is not recommended. "
                "Please use environment variables or .env file instead."
            )
        return data

    @model_validator(mode="before")
    @classmethod
    def setup_logging(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if data.get("use_rich_text", True):
                setup_logging(use_rich=True)
        return data

    @classmethod
    def get_config(
        cls,
        cfg: str | dict[str, Any] | None = None,
        overrides: dict[str, Any] | None = None,
    ):
        instance = super().get_config(cfg, overrides)
        if not isinstance(cfg, str):
            return instance
        fs = LuxonisFileSystem(cfg)
        if fs.is_mlflow:
            logger.info("Setting `project_id` and `run_id` to config's MLFlow run")
            instance.tracker.project_id = fs.experiment_id
            instance.tracker.run_id = fs.run_id
        return instance
