import logging
import sys
import warnings
from typing import Annotated, Any, Literal, TypeAlias

from luxonis_ml.enums import DatasetType
from luxonis_ml.utils import (
    BaseModelExtraForbid,
    Environ,
    LuxonisConfig,
    LuxonisFileSystem,
)
from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic.types import (
    FilePath,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveInt,
)
from typing_extensions import Self

from luxonis_train.enums import TaskType

logger = logging.getLogger(__name__)

Params: TypeAlias = dict[str, Any]


class AttachedModuleConfig(BaseModelExtraForbid):
    name: str
    attached_to: str
    alias: str | None = None
    params: Params = {}


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


class ModelNodeConfig(BaseModelExtraForbid):
    name: str
    alias: str | None = None
    inputs: list[str] = []  # From preceding nodes
    input_sources: list[str] = []  # From data loader
    freezing: FreezingConfig = FreezingConfig()
    remove_on_export: bool = False
    task: str | dict[TaskType, str] | None = None
    params: Params = {}


class PredefinedModelConfig(BaseModelExtraForbid):
    name: str
    include_nodes: bool = True
    include_losses: bool = True
    include_metrics: bool = True
    include_visualizers: bool = True
    params: Params = {}


class ModelConfig(BaseModelExtraForbid):
    name: str = "model"
    predefined_model: PredefinedModelConfig | None = None
    weights: FilePath | None = None
    nodes: list[ModelNodeConfig] = []
    losses: list[LossModuleConfig] = []
    metrics: list[MetricModuleConfig] = []
    visualizers: list[AttachedModuleConfig] = []
    outputs: list[str] = []

    @field_validator("nodes", mode="before")
    @classmethod
    def validate_nodes(cls, nodes: Any) -> Any:
        logged_general_warning = False
        if not isinstance(nodes, list):
            return nodes
        names = []
        last_body_index: int | None = None
        for i, node in enumerate(nodes):
            name = node.get("alias", node.get("name"))
            if name is None:
                raise ValueError(
                    f"Node {i} does not specify the `name` field."
                )
            if "Head" in name and last_body_index is None:
                last_body_index = i - 1
            names.append(name)
            if i > 0 and "inputs" not in node:
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
        from .predefined_models.base_predefined_model import MODELS

        if self.predefined_model:
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
            metric = self.metrics[0]
            metric.is_main_metric = True
            name = metric.alias or metric.name
            logger.info(f"Setting '{name}' as main metric.")
        else:
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
            inputs = set(
                node_name for node in self.nodes for node_name in node.inputs
            )
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
            names: set[str] = set()
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

    @model_validator(mode="before")
    @classmethod
    def check_attached_modules(cls, data: Params) -> Params:
        if "nodes" not in data:
            return data
        for section in ["losses", "metrics", "visualizers"]:
            if section not in data:
                data[section] = []
            else:
                warnings.warn(
                    f"Field `model.{section}` is deprecated. "
                    f"Please specify `{section}`under "
                    "the node they are attached to."
                )
            for node in data["nodes"]:
                if section in node:
                    cfg = node.pop(section)
                    if not isinstance(cfg, list):
                        cfg = [cfg]
                    for c in cfg:
                        c["attached_to"] = node.get("alias", node.get("name"))
                    data[section] += cfg
        return data


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
    train_view: list[str] = ["train"]
    val_view: list[str] = ["val"]
    test_view: list[str] = ["test"]
    params: Params = {}

    @field_validator("train_view", "val_view", "test_view", mode="before")
    @classmethod
    def validate_splits(cls, splits: Any) -> list[Any]:
        if isinstance(splits, str):
            return [splits]
        return splits

    @model_validator(mode="after")
    def validate_params(self) -> Self:
        dataset_type = self.params.get("dataset_type")
        if dataset_type is None:
            return self
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
    params: dict[str, Any] = {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    }


class AugmentationConfig(BaseModelExtraForbid):
    name: str
    active: bool = True
    params: Params = {}


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
                AugmentationConfig(
                    name="Normalize", params=self.normalize.params
                )
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
    params: Params = {}


class OptimizerConfig(BaseModelExtraForbid):
    name: str = "Adam"
    params: Params = {}


class SchedulerConfig(BaseModelExtraForbid):
    name: str = "ConstantLR"
    params: Params = {}


class TrainerConfig(BaseModelExtraForbid):
    preprocessing: PreprocessingConfig = PreprocessingConfig()
    use_rich_progress_bar: bool = True

    accelerator: Literal["auto", "cpu", "gpu", "tpu"] = "auto"
    devices: int | list[int] | str = "auto"
    strategy: Literal["auto", "ddp"] = "auto"
    n_sanity_val_steps: Annotated[
        int,
        Field(
            validation_alias=AliasChoices(
                "n_sanity_val_steps", "num_sanity_val_steps"
            )
        ),
    ] = 2
    profiler: Literal["simple", "advanced"] | None = None
    matmul_precision: Literal["medium", "high", "highest"] | None = None
    verbose: bool = True

    seed: int | None = None
    deterministic: bool | Literal["warn"] | None = None
    batch_size: PositiveInt = 32
    accumulate_grad_batches: PositiveInt = 1
    use_weighted_sampler: bool = False
    epochs: PositiveInt = 100
    n_workers: Annotated[
        NonNegativeInt,
        Field(validation_alias=AliasChoices("n_workers", "num_workers")),
    ] = 4
    validation_interval: Literal[-1] | PositiveInt = 5
    n_log_images: Annotated[
        NonNegativeInt,
        Field(validation_alias=AliasChoices("n_log_images", "num_log_images")),
    ] = 4
    skip_last_batch: bool = True
    pin_memory: bool = True
    log_sub_losses: bool = True
    save_top_k: Literal[-1] | NonNegativeInt = 3

    callbacks: list[CallbackConfig] = []

    optimizer: OptimizerConfig = OptimizerConfig()
    scheduler: SchedulerConfig = SchedulerConfig()

    @model_validator(mode="after")
    def validate_deterministic(self) -> Self:
        if self.seed is not None and self.deterministic is None:
            logger.warning(
                "Setting `trainer.deterministic` to True because `trainer.seed` is set. "
                "This can cause certain layers to fail. "
                "In such cases, set `trainer.deterministic` to `'warn'`."
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
                "Setting `validation_interval` same as `epochs` otherwise no checkpoint would be generated."
            )
            self.validation_interval = self.epochs
        return self


class OnnxExportConfig(BaseModelExtraForbid):
    opset_version: PositiveInt = 12
    dynamic_axes: dict[str, Any] | None = None


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
    n_trials: PositiveInt | None = 15
    timeout: PositiveInt | None = None
    storage: StorageConfig = StorageConfig()
    params: Annotated[
        dict[str, list[str | int | float | bool | list]],
        Field(default={}, min_length=1),
    ]


class Config(LuxonisConfig):
    model: Annotated[ModelConfig, Field(default_factory=ModelConfig)]
    loader: Annotated[LoaderConfig, Field(default_factory=LoaderConfig)]
    tracker: Annotated[TrackerConfig, Field(default_factory=TrackerConfig)]
    trainer: Annotated[TrainerConfig, Field(default_factory=TrainerConfig)]
    exporter: Annotated[ExportConfig, Field(default_factory=ExportConfig)]
    archiver: Annotated[ArchiveConfig, Field(default_factory=ArchiveConfig)]
    tuner: TunerConfig | None = None
    ENVIRON: Environ = Field(Environ(), exclude=True)

    @model_validator(mode="before")
    @classmethod
    def check_environment(cls, data: Any) -> Any:
        if "ENVIRON" in data:
            logger.warning(
                "Specifying `ENVIRON` section in config file is not recommended. "
                "Please use environment variables or `.env` file instead."
            )
        return data

    @classmethod
    def get_config(
        cls,
        cfg: str | dict[str, Any] | None = None,
        overrides: dict[str, Any] | list[str] | tuple[str, ...] | None = None,
    ) -> "Config":
        instance = super().get_config(cfg, overrides)
        if not isinstance(cfg, str):
            return instance
        fs = LuxonisFileSystem(cfg)
        if fs.is_mlflow:
            logger.info(
                "Setting `project_id` and `run_id` to config's MLFlow run"
            )
            instance.tracker.project_id = fs.experiment_id
            instance.tracker.run_id = fs.run_id
        return instance


def is_acyclic(graph: dict[str, list[str]]) -> bool:
    """Tests if graph is acyclic.

    @type graph: dict[str, list[str]]
    @param graph: Graph in a format of a dictionary of predecessors.
        Keys are node names, values are inputs to the node (list of node
        names).
    @rtype: bool
    @return: True if graph is acyclic, False otherwise.
    """
    graph = graph.copy()

    def dfs(node: str, visited: set[str], recursion_stack: set[str]):
        visited.add(node)
        recursion_stack.add(node)

        for predecessor in graph.get(node, []):
            if predecessor in recursion_stack:
                return True
            if predecessor not in visited:
                if dfs(predecessor, visited, recursion_stack):
                    return True

        recursion_stack.remove(node)
        return False

    visited: set[str] = set()
    recursion_stack: set[str] = set()

    for node in graph.keys():
        if node not in visited:
            if dfs(node, visited, recursion_stack):
                return False

    return True
