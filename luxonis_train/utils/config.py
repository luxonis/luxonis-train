import logging
import sys
from typing import Annotated, Any, Literal

from luxonis_ml.data import LabelType
from luxonis_ml.utils import (
    BaseModelExtraForbid,
    Environ,
    LuxonisConfig,
    LuxonisFileSystem,
)
from pydantic import Field, model_validator
from typing_extensions import Self

logger = logging.getLogger(__name__)


class AttachedModuleConfig(BaseModelExtraForbid):
    name: str
    attached_to: str
    alias: str | None = None
    params: dict[str, Any] = {}


class LossModuleConfig(AttachedModuleConfig):
    weight: float = 1.0


class MetricModuleConfig(AttachedModuleConfig):
    is_main_metric: bool = False


class FreezingConfig(BaseModelExtraForbid):
    active: bool = False
    unfreeze_after: int | float | None = None


class ModelNodeConfig(BaseModelExtraForbid):
    name: str
    alias: str | None = None
    inputs: list[str] = []  # From preceding nodes
    input_sources: list[str] = []  # From data loader
    params: dict[str, Any] = {}
    freezing: FreezingConfig = FreezingConfig()
    task: str | dict[LabelType, str] | None = None


class PredefinedModelConfig(BaseModelExtraForbid):
    name: str
    params: dict[str, Any] = {}
    include_nodes: bool = True
    include_losses: bool = True
    include_metrics: bool = True
    include_visualizers: bool = True


class ModelConfig(BaseModelExtraForbid):
    name: str = "model"
    predefined_model: PredefinedModelConfig | None = None
    weights: str | None = None
    nodes: list[ModelNodeConfig] = []
    losses: list[LossModuleConfig] = []
    metrics: list[MetricModuleConfig] = []
    visualizers: list[AttachedModuleConfig] = []
    outputs: list[str] = []

    @model_validator(mode="after")
    def check_predefined_model(self) -> Self:
        from luxonis_train.utils.registry import MODELS

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
    def check_graph(self) -> Self:
        from luxonis_train.utils.general import is_acyclic

        graph = {node.alias or node.name: node.inputs for node in self.nodes}
        if not is_acyclic(graph):
            raise ValueError("Model graph is not acyclic.")
        if not self.outputs:
            outputs: list[str] = []  # nodes which are not inputs to any nodes
            inputs = set(node_name for node in self.nodes for node_name in node.inputs)
            for node in self.nodes:
                name = node.alias or node.name
                if name not in inputs:
                    outputs.append(name)
            self.outputs = outputs
        if self.nodes and not self.outputs:
            raise ValueError("No outputs specified.")
        return self

    @model_validator(mode="after")
    def check_unique_names(self) -> Self:
        for section, objects in [
            ("nodes", self.nodes),
            ("losses", self.losses),
            ("metrics", self.metrics),
            ("visualizers", self.visualizers),
        ]:
            names = set()
            for obj in objects:
                obj: AttachedModuleConfig
                name = obj.alias or obj.name
                if name in names:
                    if obj.alias is None:
                        obj.alias = f"{name}_{obj.attached_to}"
                    if obj.alias in names:
                        raise ValueError(
                            f"Duplicate name `{name}` in `{section}` section."
                        )
                names.add(name)
        return self


class TrackerConfig(BaseModelExtraForbid):
    project_name: str | None = None
    project_id: str | None = None
    run_name: str | None = None
    run_id: str | None = None
    save_directory: str = "output"
    is_tensorboard: bool = True
    is_wandb: bool = False
    wandb_entity: str | None = None
    is_mlflow: bool = False


class LoaderConfig(BaseModelExtraForbid):
    name: str = "LuxonisLoaderTorch"
    image_source: str = "image"
    train_view: str = "train"
    val_view: str = "val"
    test_view: str = "test"
    params: dict[str, Any] = {}


class NormalizeAugmentationConfig(BaseModelExtraForbid):
    active: bool = True
    params: dict[str, Any] = {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    }


class AugmentationConfig(BaseModelExtraForbid):
    name: str
    active: bool = True
    params: dict[str, Any] = {}


class PreprocessingConfig(BaseModelExtraForbid):
    train_image_size: Annotated[
        list[int], Field(default=[256, 256], min_length=2, max_length=2)
    ] = [256, 256]
    keep_aspect_ratio: bool = True
    train_rgb: bool = True
    normalize: NormalizeAugmentationConfig = NormalizeAugmentationConfig()
    augmentations: list[AugmentationConfig] = []

    @model_validator(mode="after")
    def check_normalize(self) -> Self:
        if self.normalize.active:
            self.augmentations.append(
                AugmentationConfig(name="Normalize", params=self.normalize.params)
            )
        return self

    def get_active_augmentations(self) -> list[AugmentationConfig]:
        """Returns list of augmentations that are active.

        @rtype: list[AugmentationConfig]
        @return: Filtered list of active augmentation configs
        """
        return [aug for aug in self.augmentations if aug.active]


class CallbackConfig(BaseModelExtraForbid):
    name: str
    active: bool = True
    params: dict[str, Any] = {}


class OptimizerConfig(BaseModelExtraForbid):
    name: str = "Adam"
    params: dict[str, Any] = {}


class SchedulerConfig(BaseModelExtraForbid):
    name: str = "ConstantLR"
    params: dict[str, Any] = {}


class TrainerConfig(BaseModelExtraForbid):
    preprocessing: PreprocessingConfig = PreprocessingConfig()

    accelerator: Literal["auto", "cpu", "gpu"] = "auto"
    devices: int | list[int] | str = "auto"
    strategy: Literal["auto", "ddp"] = "auto"
    num_sanity_val_steps: int = 2
    profiler: Literal["simple", "advanced"] | None = None
    matmul_precision: Literal["medium", "high", "highest"] | None = None
    verbose: bool = True

    seed: int | None = None
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
    def check_num_workes_platform(self) -> Self:
        if (
            sys.platform == "win32" or sys.platform == "darwin"
        ) and self.num_workers != 0:
            self.num_workers = 0
            logger.warning(
                "Setting `num_workers` to 0 because of platform compatibility."
            )
        return self

    @model_validator(mode="after")
    def check_validation_interval(self) -> Self:
        if self.validation_interval > self.epochs:
            logger.warning(
                "Setting `validation_interval` same as `epochs` otherwise no checkpoint would be generated."
            )
            self.validation_interval = self.epochs
        return self


class OnnxExportConfig(BaseModelExtraForbid):
    opset_version: int = 12
    dynamic_axes: dict[str, Any] | None = None


class BlobconverterExportConfig(BaseModelExtraForbid):
    active: bool = False
    shaves: int = 6
    version: Literal["2021.2", "2021.3", "2021.4", "2022.1", "2022.3_RVC3"] = "2022.1"


class ArchiveConfig(BaseModelExtraForbid):
    name: str | None = None
    upload_to_run: bool = True
    upload_url: str | None = None


class ExportConfig(ArchiveConfig):
    name: str | None = None
    input_shape: list[int] | None = None
    data_type: Literal["int8", "fp16", "fp32"] = "fp16"
    reverse_input_channels: bool = True
    scale_values: list[float] | None = None
    mean_values: list[float] | None = None
    output_names: list[str] | None = None
    onnx: OnnxExportConfig = OnnxExportConfig()
    blobconverter: BlobconverterExportConfig = BlobconverterExportConfig()

    @model_validator(mode="after")
    def check_values(self) -> Self:
        def pad_values(values: float | list[float] | None):
            if values is None:
                return None
            if isinstance(values, float):
                return [values] * 3

        self.scale_values = pad_values(self.scale_values)
        self.mean_values = pad_values(self.mean_values)
        return self


class StorageConfig(BaseModelExtraForbid):
    active: bool = True
    storage_type: Literal["local", "remote"] = "local"


class TunerConfig(BaseModelExtraForbid):
    study_name: str = "test-study"
    continue_existing_study: bool = True
    use_pruner: bool = True
    n_trials: int | None = 15
    timeout: int | None = None
    storage: StorageConfig = StorageConfig()
    params: Annotated[
        dict[str, list[str | int | float | bool | list]],
        Field(default={}, min_length=1),
    ]


class Config(LuxonisConfig):
    model: ModelConfig = ModelConfig()
    loader: LoaderConfig = LoaderConfig()
    tracker: TrackerConfig = TrackerConfig()
    trainer: TrainerConfig = TrainerConfig()
    exporter: ExportConfig = ExportConfig()
    archiver: ArchiveConfig = ArchiveConfig()
    tuner: TunerConfig | None = None
    ENVIRON: Environ = Field(Environ(), exclude=True)

    @model_validator(mode="before")
    @classmethod
    def check_environment(cls, data: Any) -> Any:
        if "ENVIRON" in data:
            logger.warning(
                "Specifying `ENVIRON` section in config file is not recommended. "
                "Please use environment variables or .env file instead."
            )
        return data

    @classmethod
    def get_config(
        cls,
        cfg: str | dict[str, Any] | None = None,
        overrides: dict[str, Any] | list[str] | tuple[str, ...] | None = None,
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
