"""The schema of the configuration file.

Every section of the YAML file is a model here, and `Config` is the
root. The models reject an unknown key, so a typo fails at load time
instead of being ignored.

"""

import json
import re
import sys
from collections.abc import Mapping, Sequence
from contextlib import suppress
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal, NamedTuple, cast

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
    field_serializer,
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
from luxonis_train.registry import NODES
from luxonis_train.upgrade import upgrade_config

if TYPE_CHECKING:
    from luxonis_train.config.predefined_models import BasePredefinedModel


class ImageSize(NamedTuple):
    """The height and the width of an image, in pixels."""

    height: int
    width: int


class AttachedModuleConfig(ConfigItem):
    """The fields every attached module shares.

    Attributes:
        alias: A name for this module in the logs. Two modules of the
            same class on one node need one.

    """

    alias: str | None = None

    @property
    def identifier(self) -> str:
        return self.alias or self.name


class LossModuleConfig(AttachedModuleConfig):
    """A loss attached to a node.

    ``name`` is the class name of a registered loss and ``params``
    reaches its constructor. See
    `luxonis_train.attached_modules.losses`. At least one node in a
    config must carry a loss.

    Attributes:
        weight: The factor this loss contributes to the total. A weight
            of 0 removes the loss from training.

    """

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
    """A metric attached to a node.

    ``name`` is the class name of a registered metric and ``params``
    reaches its constructor. See
    `luxonis_train.attached_modules.metrics`.

    Attributes:
        is_main_metric: Track checkpoints on this metric. Only one
            metric in a config can set it.

    """

    is_main_metric: bool = False


class FreezingConfig(BaseModelExtraForbid):
    """Whether a node trains, and when it starts.

    Attributes:
        active: Freeze the node, so its weights do not update.
        unfreeze_after: When to unfreeze. An integer counts epochs, and
            a fraction takes that share of the training.
        lr_after_unfreeze: The base learning rate for the groups of this
            node from the unfreeze epoch on.

            It replaces the ``lr`` of the group, its ``initial_lr``, and
            the entry of the scheduler, and the scheduler continues from
            its current position with the new base. Left out, the groups
            resume at whatever value the scheduler has reached.

    """

    active: bool = False
    unfreeze_after: NonNegativeInt | NonNegativeFloat | None = None
    lr_after_unfreeze: NonNegativeFloat | None = None


class ParameterPattern(BaseModelExtraForbid):
    """A pattern that selects the parameters of a node.

    ``re.search`` matches both fields, without anchors and without case.
    ``module_type: Linear`` therefore also claims ``LazyLinear`` and the
    ``NonDynamicallyQuantizableLinear`` inside
    ``nn.MultiheadAttention``, and ``name: conv1`` also claims
    ``branch1.conv10.weight``. Anchor the pattern, as in ``module_type:
    ^Linear$``, when you mean an exact match.

    Attributes:
        name: A pattern matched against the name of a parameter.
        module_type: A pattern matched against the class name of the
            module that owns a parameter.

    """

    name: str | None = None
    module_type: str | None = None

    @model_validator(mode="after")
    def validate_pattern(self) -> Self:
        if self.name is None and self.module_type is None:
            raise ValueError(
                "At least one of `name` or `module_type` must be specified for parameter pattern."
            )
        if self.name == "":
            raise ValueError("Parameter pattern `name` cannot be empty.")
        if self.module_type == "":
            raise ValueError(
                "Parameter pattern `module_type` cannot be empty."
            )
        return self

    def matches(self, module_type: str, parameter_name: str) -> bool:
        if self.name is not None and not re.search(
            self.name, parameter_name, flags=re.IGNORECASE
        ):
            return False
        return self.module_type is None or bool(
            re.search(self.module_type, module_type, flags=re.IGNORECASE)
        )


class SchedulerConfig(ConfigItem):
    """The scheduler to use.

    ``name`` is the class name of any scheduler of
    ``torch.optim.lr_scheduler``, and ``params`` reaches its
    constructor.

    """

    name: str = "ConstantLR"

    def get_sequential_lr_params(self) -> "SequentialLRParams":
        if self.name != "SequentialLR":
            raise RuntimeError(
                f"Scheduler '{self.name}' is not 'SequentialLR'. "
                "Cannot get `SequentialLR` parameters."
            )

        if "schedulers" not in self.params or "milestones" not in self.params:
            raise ValueError(
                "SequentialLR requires 'schedulers' and 'milestones' parameters."
            )
        return SequentialLRParams(**self.params)  # type: ignore

    def to_finetuning(self) -> "FinetuningSchedulerConfig":
        return FinetuningSchedulerConfig(name=self.name, params=self.params)


class SequentialLRParams(BaseModelExtraForbid):
    """The parameters ``SequentialLR`` requires.

    Attributes:
        schedulers: The schedulers to run, in order.
        milestones: The epochs at which one scheduler hands over to the
            next.
        last_epoch: The epoch to resume from.

    """

    schedulers: list[SchedulerConfig]
    milestones: list[int]
    last_epoch: int = -1


class FinetuningSchedulerConfig(SchedulerConfig):
    """A scheduler override, whose ``name`` may be left out.

    An override that omits ``name`` inherits the name of the trainer-
    level scheduler and merges into its ``params``. An override that
    names a different scheduler drops those ``params`` and uses only its
    own.

    """

    name: str | None = None


class OptimizerConfig(ConfigItem):
    """The optimizer to use.

    ``name`` is the class name of any optimizer of ``torch.optim``, and
    ``params`` reaches its constructor.

    """

    name: str = "Adam"

    def to_finetuning(self) -> "FinetuningOptimizerConfig":
        return FinetuningOptimizerConfig(name=self.name, params=self.params)


class FinetuningOptimizerConfig(OptimizerConfig):
    """An optimizer override, whose ``name`` may be left out.

    An override that omits ``name`` inherits the name of the trainer-
    level optimizer and merges into its ``params``. An override that
    names a different optimizer drops those ``params`` and uses only its
    own.

    """

    name: str | None = None


class FinetuningConfig(BaseModelExtraForbid):
    """A different optimizer or scheduler for part of a node.

    Use it to train a pretrained backbone at a lower learning rate than
    a fresh head.

    The trainer evaluates the entries of a node in order, and the first
    entry whose pattern matches a parameter claims it. Put the specific
    rules first and the general ones last.

    Entries that share an optimizer name, a scheduler name, and the
    scheduler ``params`` collapse into one optimizer with one parameter
    group for each entry, so a different ``lr`` between entries still
    applies. Any other difference produces a separate inner optimizer,
    and one `CompositeOptimizer
    <luxonis_train.optimizers.composite_optimizer.CompositeOptimizer>`
    drives them all. Training therefore stays in the automatic
    optimization of Lightning, and gradient accumulation and gradient
    clipping keep working.

    Every parameter reaches an optimizer. A parameter that no entry
    claims falls to a default group that uses the trainer-level
    optimizer and scheduler. A frozen parameter stays in its group and
    is skipped while it is frozen, so a node that unfreezes mid-training
    already has an optimizer waiting.

    Every parameter group carries a name, so `lightning.pytorch.callbacks.LearningRateMonitor` logs
    one series for each group instead of ``pg1``, ``pg2``, and so on. A
    finetuning entry is named ``<node>/<index>``, a strategy group
    ``strategy/<tag>``, and the default group ``default``. A group that
    has to be scoped to one node gains the node name, as in
    ``default/<node>``. A configuration that produces a single group
    leaves it unnamed.

    Attributes:
        parameters: The parameters this entry claims. Left out, the
            entry claims every parameter of the node. A plain string is
            short for a pattern on the parameter name.
        optimizer: The optimizer for the claimed parameters.
        scheduler: The scheduler for the claimed parameters.

    """

    parameters: list[ParameterPattern] | None = None
    optimizer: FinetuningOptimizerConfig | None = None
    scheduler: FinetuningSchedulerConfig | None = None

    @field_validator("parameters", mode="before")
    @classmethod
    def validate_parameters(cls, value: Any) -> Any:
        parsed_patterns = []
        if isinstance(value, str | dict | ParameterPattern):
            value = [value]
        if not isinstance(value, list):
            return value
        if not value:
            raise ValueError(
                "`parameters` must contain at least one parameter pattern."
            )
        for item in value:
            if isinstance(item, str):
                parsed_patterns.append(ParameterPattern(name=item))
            elif isinstance(item, dict):
                parsed_patterns.append(ParameterPattern(**item))
            elif isinstance(item, ParameterPattern):
                parsed_patterns.append(item)
            else:
                raise TypeError(
                    "Parameter patterns must be strings, dictionaries, "
                    "or ParameterPattern instances."
                )
        return parsed_patterns


class NodeConfig(ConfigItem):
    """One node of the model graph.

    ``name`` is the class name of a registered node and ``params``
    reaches its constructor. See `luxonis_train.nodes` for the nodes.

    Attributes:
        alias: A name for this node in the graph. Other nodes refer to
            the alias, and the weights bind to it.
        inputs: The nodes that feed this one. Left empty on the first
            node, the node reads from the loader.
        input_sources: The loader outputs that feed this node, when it
            reads the data directly.
        remove_on_export: Drop this node from the exported model.
        task_name: The task of the dataset this head reads. Set it when
            a dataset carries more than one task.
        metadata_task_override: The dataset metadata field this head
            reads, when its name differs from the default.
        variant: The variant of the node. Each node docstring lists the
            variants it declares.
        losses: The losses attached to this node.
        metrics: The metrics attached to this node.
        visualizers: The visualizers attached to this node.
        finetuning: Optimizer and scheduler overrides for this node.
        freezing: Whether this node trains.

    """

    alias: str | None = None
    inputs: list[str] = []  # From preceding nodes
    input_sources: list[str] = []  # From data loader
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
    finetuning: list[FinetuningConfig] = []
    freezing: FreezingConfig = Field(default_factory=FreezingConfig)

    @field_validator("finetuning", mode="before")
    @classmethod
    def validate_finetuning(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return [value]
        return value

    @property
    def identifier(self) -> str:
        return self.alias or self.name


class PredefinedModelConfig(ConfigItem):
    """A reference to a predefined model, with its parameters.

    ``name`` selects the model and ``params`` reaches its constructor.
    See `luxonis_train.config.predefined_models` for the models and the
    parameters each one accepts.

    Attributes:
        variant: The variant to build. Most models offer ``light``,
            ``medium``, and ``heavy``.
        version: The version of the model. ``latest`` follows the newest
            one, and an integer pins the graph a config was written
            against.
        include_losses: Add the losses of the model.
        include_metrics: Add the metrics of the model.
        include_visualizers: Add the visualizers of the model.

    """

    variant: str | Literal["default", "none"] | None = "default"
    version: int | Literal["latest"] = "latest"
    include_losses: bool = True
    include_metrics: bool = True
    include_visualizers: bool = True


class ModelConfig(BaseModelExtraForbid):
    """The model graph, or the predefined model that generates one.

    Build the graph by hand with ``nodes``, or name a predefined model
    and let it contribute the nodes. You can do both: the nodes a
    predefined model generates are appended to the ones you list.

    Attributes:
        name: The name of the model. It names the output directory and
            the exported files.
        predefined_model: A predefined model that generates the nodes.
        weights: A checkpoint to start from. ``trainer.resume_training``
            decides whether the optimizer state comes with it.
        nodes: The nodes of the graph.
        outputs: The nodes whose outputs the exported model returns.
            Left empty, it holds the nodes that feed no other node.

    """

    name: str = "model"
    predefined_model: Annotated[
        PredefinedModelConfig | None, Field(exclude=True)
    ] = None
    weights: Annotated[FilePath | None, Field(exclude=True)] = None
    nodes: list[NodeConfig] = []
    outputs: list[str] = []

    @field_validator("nodes", mode="before")
    @classmethod
    def validate_nodes(cls, nodes: ParamValue) -> Any:
        if not check_type(nodes, list[dict]):
            return nodes

        return cls._populate_implicit_node_inputs(nodes)

    @staticmethod
    def _populate_implicit_node_inputs(
        nodes: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        logged_general_warning = False
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
            if i == 0 or "inputs" in node or "input_sources" in node:
                continue

            previous_index = (
                last_body_index if last_body_index is not None else i - 1
            )
            prev_name = names[previous_index]
            if not logged_general_warning:
                logger.warning(
                    f"Field `inputs` not specified for node '{name}'. "
                    "Assuming the model follows a linear multi-head topology "
                    "(backbone -> (neck?) -> head1, head2, ...). "
                    "If this is incorrect, please specify the `inputs` field explicitly."
                )
                logged_general_warning = True

            logger.warning(f"Setting `inputs` of '{name}' to '{prev_name}'. ")
            node["inputs"] = [prev_name]
        return nodes

    @model_validator(mode="after")
    def validate_predefined_model(self) -> Self:
        if self.predefined_model is None:
            return self

        from luxonis_train.config.predefined_versions import (
            resolve_predefined_class,
            resolved_class_name,
        )

        cls = resolve_predefined_class(
            self.predefined_model.name, self.predefined_model.version
        )
        resolved = resolved_class_name(
            self.predefined_model.name, self.predefined_model.version
        )
        message = f"Using predefined model: `{self.predefined_model.name}`"
        if resolved != self.predefined_model.name:
            message += f" (resolved to `{resolved}`)"
        logger.info(message)
        kwargs: dict[str, Any] = dict(self.predefined_model.params or {})
        if not kwargs.get("variant"):
            kwargs["variant"] = self.predefined_model.variant
        model = cast("BasePredefinedModel", cls(**kwargs))
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
            inputs = {
                node_name for node in self.nodes for node_name in node.inputs
            }
            self.outputs = [
                node.alias or node.name
                for node in self.nodes
                if (node.alias or node.name) not in inputs
            ]
        if self.nodes and not self.outputs:
            raise ValueError("No outputs specified.")
        return self

    @model_validator(mode="after")
    def check_for_invalid_characters(self) -> Self:
        for node in self.nodes:
            for module in self._node_modules(node):
                self._validate_module_characters(module)

        return self

    @model_validator(mode="after")
    def check_unique_names(self) -> Self:
        for node in self.nodes:
            self._make_node_module_names_unique(node, node.losses)
            self._make_node_module_names_unique(node, node.metrics)
            self._make_node_module_names_unique(node, node.visualizers)
        return self

    @staticmethod
    def _node_modules(
        node: NodeConfig,
    ) -> list[AttachedModuleConfig | NodeConfig]:
        return [node, *node.losses, *node.metrics, *node.visualizers]

    @staticmethod
    def _validate_module_characters(
        module: AttachedModuleConfig | NodeConfig,
    ) -> None:
        invalid_parts = [
            f"{field} '{value}'"
            for field, value in (
                ("alias", module.alias),
                ("name", module.name),
            )
            if value and "/" in value
        ]
        if invalid_parts:
            raise ValueError(
                f"The {', '.join(invalid_parts)} contain a '/', which is not allowed. "
                "Please rename to remove any '/' characters."
            )

    @staticmethod
    def _make_node_module_names_unique(
        node: NodeConfig,
        modules: Sequence[AttachedModuleConfig],
    ) -> None:
        names: set[str] = set()
        node_index = 0
        for module in [node, *modules]:
            name = module.alias or module.name
            if name not in names:
                names.add(name)
                continue

            if module.alias is None:
                module.alias = (
                    module.name
                    if isinstance(module, NodeConfig)
                    else f"{name}_{node.alias}"
                )

            if module.alias in names:
                new_alias = f"{module.alias}_{node_index}"
                logger.warning(
                    f"Duplicate name: {module.alias}. Renaming to {new_alias}."
                )
                module.alias = new_alias
                node_index += 1

            names.add(name)

    @property
    def head_nodes(self) -> list[NodeConfig]:
        from luxonis_train.nodes import BaseHead

        return [
            node
            for node in self.nodes
            if issubclass(NODES._module_dict.get(node.name, object), BaseHead)
        ]


class TrackerConfig(BaseModelExtraForbid):
    """Where the metrics, the images, and the checkpoints go.

    More than one backend can be active at once.

    Attributes:
        project_name: The project the run belongs to.
        project_id: The project identifier, which MLFlow uses instead of
            the name.
        run_name: The name of the run. Left out, it is generated.
        run_id: An existing MLFlow run to continue.
        save_directory: Where the logs, the checkpoints, and the
            exported files go.
        is_tensorboard: Log to TensorBoard.
        is_wandb: Log to Weights and Biases.
        wandb_entity: The Weights and Biases entity that owns the run.
        is_mlflow: Log to MLFlow.

    """

    project_name: str | None = None
    project_id: str | None = None
    run_name: str | None = None
    run_id: str | None = None
    save_directory: Annotated[Path, Field(exclude=True)] = Path("output")
    is_tensorboard: bool = True
    is_wandb: bool = False
    wandb_entity: str | None = None
    is_mlflow: bool = False


class LoaderConfig(ConfigItem):
    """Where the data comes from, and which splits each stage reads.

    ``name`` is the class name of a registered loader and ``params``
    reaches its constructor. See `luxonis_train.loaders`.

    Attributes:
        name: The class name of a registered loader. See
            `luxonis_train.loaders`.
        image_source: The name of the input group that holds the image.
        train_view: The dataset splits to train on.
        val_view: The dataset splits to validate on.
        test_view: The dataset splits to test on.

    """

    name: str = "LuxonisLoaderTorch"
    image_source: str = "image"
    train_view: list[str] = ["train"]
    val_view: list[str] = ["val"]
    test_view: list[str] = ["test"]

    @field_serializer("params")
    def serialize_params(self, info: SerializationInfo) -> Any:
        data = self.params.copy()
        if self.name == "DummyLoader":
            data.pop("n_classes", None)
            data.pop("n_keypoints", None)
            data.pop("class_names", None)
        return data

    @field_serializer("name")
    def serialize_name(self, info: SerializationInfo) -> str:
        if self.name == "DummyLoader":
            return "LuxonisLoaderTorch"
        return self.name

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
    """The normalization applied after every other augmentation.

    The default ``params`` hold the ImageNet mean and standard
    deviation. Naming ``Normalize`` in ``augmentations`` instead
    overrides this section.

    Attributes:
        active: Normalize the images.
        params: The parameters of the ``Albumentations`` ``Normalize``
            transform.

    """

    active: bool = True
    params: Params = {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    }


class AugmentationConfig(ConfigItem):
    """One augmentation.

    ``name`` is the name of an ``Albumentations`` transform, and
    ``params`` reaches its constructor.

    ``VerticalFlip`` and ``HorizontalFlip`` do not reorder the keypoint
    indices. Use them only when no class of the dataset has a
    symmetrical counterpart, such as a left arm and a right arm.

    Attributes:
        active: Apply this augmentation.
        use_for_resizing: Resize with this augmentation instead of the
            default one.

            The ``height`` and the ``width`` are overridden with
            ``train_image_size``, and ``keep_aspect_ratio`` is ignored.
            With a probability below 1, the default resize covers the
            remaining images, so an image is always resized.
        apply_on_stages: The stages that apply this augmentation.

    """

    active: bool = True
    use_for_resizing: bool = False
    apply_on_stages: list[Literal["train", "val", "test"]] = Field(
        default_factory=lambda: ["train"]
    )


class PreprocessingConfig(BaseModelExtraForbid):
    """The resizing and the augmentations applied to every image.

    The augmentations come from `Albumentations
    <https://albumentations.ai/docs/>`_, and any pixel-level or
    spatial-level transform of that library can be named here. The batch
    augmentations ``Mosaic4`` and ``MixUp`` are also available.

    Attributes:
        train_image_size: The size every image is resized to, as height
            and width.
        keep_aspect_ratio: Pad the image instead of stretching it.
        color_space: The colour space the model trains on.
        normalize: The normalization applied to every image.
        augmentations: The augmentations, in the order they apply.

    """

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

    @model_validator(mode="after")
    def check_use_for_resizing(self) -> Self:
        train_h, train_w = self.train_image_size
        for aug in self.augmentations:
            if not aug.use_for_resizing:
                continue

            aug_h = aug.params.get("height")
            aug_w = aug.params.get("width")
            aug.params.setdefault("p", 1.0)
            if aug_h != train_h or aug_w != train_w:
                logger.warning(
                    f"Augmentation '{aug.name}' is marked as 'use_for_resizing' "
                    f"but its (height, width) doesn't match "
                    f"train_image_size ({train_h}, {train_w}). "
                    f"Overriding to match train_image_size."
                )
            aug.params["height"] = train_h
            aug.params["width"] = train_w

            if self.keep_aspect_ratio and aug.params["p"] == 1:
                logger.warning(
                    f"Augmentation '{aug.name}' is marked as 'use_for_resizing'. "
                    f"The 'keep_aspect_ratio' preprocessing parameter is ignored "
                    f"when a custom resizing augmentation is used."
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

    def get_active_augmentations(self) -> list[AugmentationConfig]:
        """Get a list of augmentations that are active.

        Returns:
            list[AugmentationConfig]: Filtered list of active augmentation
            configs.

        """
        return [
            AugmentationConfig(
                name=aug.name,
                params=aug.params,
                use_for_resizing=aug.use_for_resizing,
                apply_on_stages=aug.apply_on_stages,
            )
            for aug in self.augmentations
            if aug.active
        ]


class CallbackConfig(ConfigItem):
    """One callback.

    ``name`` is the class name of a registered callback and ``params``
    reaches its constructor. See `luxonis_train.callbacks` for the
    callbacks, and for the ones that are added automatically.

    Attributes:
        active: Run this callback.

    """

    active: bool = True


class TrainerConfig(BaseModelExtraForbid):
    """Everything about how the model trains.

    The trainer calls the callbacks in the order this section lists
    them.

    Attributes:
        preprocessing: The resizing and the augmentations.
        precision: The precision of the training. ``16-mixed`` is much
            faster on a GPU that supports it.
        accelerator: The hardware to train on.
        devices: How many devices to use, or which ones.
        strategy: The distribution strategy.
        n_sanity_val_steps: How many validation batches to run before
            training starts.
        profiler: The Lightning profiler, which reports where the time
            goes.
        matmul_precision: The internal precision of a float32 matrix
            multiplication.
        seed: The seed of every random number generator. Set it to make
            a run reproducible.
        n_validation_batches: How many validation and test batches to
            evaluate.

            A positive number takes the first batches of each view, and
            ``-1`` takes them in full. Left out,
            ``smart_cfg_auto_populate`` may set it.
        deterministic: Use the deterministic kernels of PyTorch. Some
            layers have none, and ``warn`` lets those through.
        smart_cfg_auto_populate: Fill in the fields a config leaves out,
            and log what was filled.

            The rules are:

            - With no ``training_strategy``, no ``optimizer``, and no
              ``scheduler``, the trainer uses ``Adam`` and ``ConstantLR``.
            - ``CosineAnnealingLR`` without ``T_max`` gets ``epochs``.
            - ``Mosaic4`` without ``out_width`` and ``out_height`` gets
              ``train_image_size``.
            - When ``train_view``, ``val_view``, and ``test_view`` are the
              same and ``n_validation_batches`` is unset, it becomes 10,
              so validation does not run over the whole training set. An
              explicit value, ``-1`` included, is kept.
            - A predefined model gets ``accumulate_grad_batches`` of
              ``int(64 / batch_size)``, a gradient accumulation schedule
              of ``{0: 1, 1: (1 + accumulate_grad_batches) // 2, 2:
              accumulate_grad_batches}``, and its loss weights scaled by
              that factor.
            - ``ConvertOnTrainEnd``, ``TestOnTrainEnd``, and
              ``UploadCheckpoint`` are added.
        batch_size: How many samples one step uses.
        accumulate_grad_batches: How many batches to accumulate before a
            step. It raises the effective batch size without the memory.
        gradient_clip_val: The value to clip the gradients at. It can
            stop the gradients from exploding.
        gradient_clip_algorithm: Clip the gradients by their norm, or
            element by element.
        use_weighted_sampler: Sample the rare classes more often. It
            works only on a classification task.
        epochs: How many epochs to train.
        overfit_batches: Train and validate on this many batches only.

            Use it to check that a config learns at all, or to test a
            visualizer. Training and validation each keep their batches
            across epochs but draw from different data. Set ``seed`` to
            make the choice repeatable.
        resume_training: Continue the run that ``model.weights`` came
            from.

            The optimizer, the scheduler, and the epoch count all
            continue, so ``epochs`` must exceed the epochs already
            trained. Left false, only the weights load and the training
            state starts fresh, which is what a finetuning run wants.
        strict_weights_loading: Require the checkpoint to match the
            model exactly. The keys of a loss, a metric, or a visualizer
            are still allowed to be missing.
        n_workers: How many worker processes load the data. It is forced
            to 0 on Windows and macOS.
        validation_interval: How many epochs pass between two validation
            runs.
        run_validation_after_first_epoch: Also validate after the first
            epoch, which shows early whether a run is broken.
        n_log_images: How many images each head contributes to the
            logged visualizations.
        skip_last_batch: Drop the last batch of an epoch when it is
            smaller than the others.
        pin_memory: Pin the memory of the data loader, which speeds up
            the transfer to a GPU.
        log_sub_metrics: Log the parts of a metric, such as
            ``map_small`` beside ``map``.
        log_sub_losses: Log the parts of a loss beside the total.
        save_top_k: How many checkpoints to keep. ``-1`` keeps every
            one.
        callbacks: The callbacks to run, in order.
        optimizer: The optimizer, for the parameters that no finetuning
            rule and no strategy claims.
        scheduler: The scheduler, for the parameters that no finetuning
            rule and no strategy claims.
        training_strategy: A strategy that owns the optimization
            schedule. See `luxonis_train.strategies`.

    """

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
    overfit_batches: NonNegativeInt = 0
    resume_training: bool = False
    strict_weights_loading: bool = False
    n_workers: NonNegativeInt = 4
    validation_interval: Literal[-1] | PositiveInt = 5
    run_validation_after_first_epoch: bool = False
    n_log_images: NonNegativeInt = 4
    skip_last_batch: bool = True
    pin_memory: bool = True
    log_sub_metrics: bool = True
    log_sub_losses: bool = True
    save_top_k: Literal[-1] | NonNegativeInt = 3

    callbacks: list[CallbackConfig] = []

    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)

    training_strategy: ConfigItem | None = None

    @model_validator(mode="after")
    def validate_gradient_acc_scheduler(self) -> Self:
        """Keys in the GradientAccumulationScheduler.params.scheduling
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

            callback.params["scheduling"] = cast(
                ParamValue,
                {
                    int(k) if isinstance(k, str) and k.isdigit() else k: v
                    for k, v in scheduling.items()
                },
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
    def validate_overfit_batches(self) -> Self:
        if self.overfit_batches > 0 and self.seed is None:
            logger.warning(
                "Using `overfit_batches` without setting `seed` may cause "
                "different batches to be selected each run due to shuffling. "
                "Consider setting `trainer.seed` for reproducible results."
            )
        return self

    @model_validator(mode="after")
    def check_n_workers_platform(self) -> Self:
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
        needs to be updated before other callbacks.
        """
        self.callbacks.sort(key=lambda v: 0 if v.name == "EMACallback" else 1)
        return self

    @model_validator(mode="after")
    def check_convert_callbacks(self) -> Self:
        callback_names = {cb.name for cb in self.callbacks if cb.active}
        has_convert = "ConvertOnTrainEnd" in callback_names
        has_export = "ExportOnTrainEnd" in callback_names
        has_archive = "ArchiveOnTrainEnd" in callback_names

        if has_convert and (has_export or has_archive):
            redundant = []
            for cb in self.callbacks:
                if (
                    cb.name in ("ExportOnTrainEnd", "ArchiveOnTrainEnd")
                    and cb.active
                ):
                    cb.active = False
                    redundant.append(cb.name)
            if redundant:
                logger.warning(
                    f"Deactivated {redundant} because 'ConvertOnTrainEnd' is active "
                    "and already includes export and archive functionality."
                )
        elif has_export and has_archive:
            logger.warning(
                "Both 'ExportOnTrainEnd' and 'ArchiveOnTrainEnd' callbacks are set. "
                "Consider using 'ConvertOnTrainEnd' instead, which combines both "
                "and also handles platform-specific conversions (blobconverter/HubAI SDK)."
            )
        return self


class OnnxExportConfig(BaseModelExtraForbid):
    """The options of the ONNX export.

    Attributes:
        opset_version: The ONNX opset to target.
        dynamic_axes: The axes that stay dynamic in the exported model.
        disable_onnx_simplification: Keep the graph as exported, without
            simplification.
        unique_onnx_initializers: Rename the initializers so each block
            owns unique names.

    """

    opset_version: PositiveInt = 16
    dynamic_axes: Params | None = None
    disable_onnx_simplification: bool = False
    unique_onnx_initializers: bool = False


class BlobconverterExportConfig(BaseModelExtraForbid):
    """Conversion to the ``.blob`` format.

    ``blobconverter`` is deprecated and only converts for RVC2. Use
    `HubAIExportConfig` instead.

    Attributes:
        active: Convert to ``.blob``.
        shaves: How many SHAVE cores the blob targets.
        version: The OpenVINO version to convert with.

    """

    active: bool = False
    shaves: int = 6
    version: Literal["2021.2", "2021.3", "2021.4", "2022.1", "2022.3_RVC3"] = (
        "2022.1"
    )


class HubAIExportConfig(BaseModelExtraForbid):
    """Conversion through the `HubAI SDK
    <https://github.com/luxonis/hubai-sdk>`_.

    This is the supported way to convert a model for a device. It needs
    the ``HUBAI_API_KEY`` environment variable.

    The name of the model, the dataset, and the input shape together
    decide whether the upload creates a new model, a new variant, or a
    new version of a variant. The `Luxonis documentation
    <https://docs.luxonis.com/cloud/hubai/model-registry/concepts/>`_
    describes the difference.

    Attributes:
        active: Convert through the HubAI SDK.
        platform: The device to convert for. It is required when
            ``active`` is true.
        params: Extra arguments for the conversion.
        delete_remote_model: Delete the uploaded variant after the
            conversion.

    """

    active: bool = False
    platform: Literal["rvc2", "rvc3", "rvc4", "hailo"] | None = None
    params: Params = Field(default_factory=dict)
    delete_remote_model: bool = False

    @model_validator(mode="after")
    def validate_platform(self) -> Self:
        if self.active and self.platform is None:
            raise ValueError(
                "The `platform` field is required when `hubai.active` is True. "
                "Please specify a target platform: 'rvc2', 'rvc3', 'rvc4'."
            )
        if self.platform == "hailo":
            raise NotImplementedError(
                "Hailo platform conversion is not yet supported."
            )
        return self


class ArchiveConfig(BaseModelExtraForbid):
    """How the NN Archive is named and uploaded.

    Attributes:
        name: The name of the archive. Left out, the model name is used.
        upload_to_run: Attach the files to the tracked run.
        upload_url: A location to upload the files to.

    """

    name: str | None = None
    upload_to_run: bool = True
    upload_url: str | None = None


def _validate_quantization_mode(value: str) -> str:
    value = value.upper()

    shorthand_map = {
        "FP16": "FP16_STANDARD",
        "FP32": "FP32_STANDARD",
    }
    # values are taken from hubai_sdk enum definition:
    # hubai_sdk.utils.hubai_models.EnumQuantizationMode
    valid_modes = (
        "INT8_STANDARD",
        "INT8_ACCURACY_FOCUSED",
        "INT8_INT16_MIXED",
        "INT8_INT16_MIXED_ACCURACY_FOCUSED",
        "FP16_STANDARD",
        "FP32_STANDARD",
    )

    value = shorthand_map.get(value, value)

    if value not in valid_modes:
        raise ValueError(
            f"Invalid quantization_mode: '{value}'. "
            f"Valid options are: {sorted(valid_modes)}"
        )
    return value


class AdaroundConfig(BaseModelExtraForbid):
    """Adaptive rounding of the weights during quantization.

    AIMET rounds a weight to the nearest quantized value by default.
    AdaRound instead learns the rounding from the calibration data,
    which usually raises the accuracy of the quantized model.

    Attributes:
        active: Learn the rounding of the weights.
        default_num_iterations: How many iterations to optimize for.
            Left out, AIMET uses 10000 at 8 bits or more, and 15000
            below.
        default_reg_param: The trade-off between the rounding loss and
            the reconstruction loss.
        default_beta_range: The start and the end of the beta annealing.
        default_warm_start: The share of the iterations during which the
            rounding loss has no effect.

    """

    active: bool = False
    default_num_iterations: PositiveInt | None = None
    default_reg_param: float = 0.01
    default_beta_range: tuple[int, int] = (20, 2)
    default_warm_start: float = 0.2


class AIMETConfig(BaseModelExtraForbid):
    """Quantization with `AIMET
    <https://quic.github.io/aimet-pages/releases/latest/index.html>`_.

    The advanced techniques are slow. AdaRound alone can take from 40
    minutes to several hours, depending on the model and the amount of
    calibration data.

    Attributes:
        active: Quantize with AIMET.
        default_output_bw: The bit width of the activations.
        default_param_bw: The bit width of the parameters.
        default_data_type: The data type of a quantized value.
        quant_scheme: How the quantization ranges are chosen.
        config: Extra AIMET settings, inline or as a path to a JSON
            file. See the `AIMET documentation
            <https://quic.github.io/aimet-pages/releases/latest/techniques/runtime_config.html>`_.
        max_calibration_images: How many validation images calibrate the
            quantization. Left out, the whole validation split is used.
            It is independent of ``n_validation_batches``.
        fold_batch_norms: Fold the batch normalization layers into the
            convolutions first.
        cross_layer_equalization: Balance the weight ranges across
            consecutive layers first.
        batch_norm_reestimation: Re-estimate the batch normalization
            statistics after quantization.
        sequential_mse: Optimize the quantization of each layer against
            the output of the float model.
        adaround: Adaptive rounding of the weights.
        epochs: How many epochs of quantization-aware training to run.
        optimizer: The optimizer of the quantization-aware training.
        scheduler: The scheduler of the quantization-aware training.

    """

    active: bool = False

    default_output_bw: Literal[4, 8, 16] = 8
    default_param_bw: Literal[4, 8, 16] = 8
    default_data_type: Literal["int", "float"] = "int"
    quant_scheme: Literal["min_max", "tf", "tf_enhanced"] = "min_max"
    config: Params | None = None
    max_calibration_images: PositiveInt | None = None

    fold_batch_norms: bool = False
    cross_layer_equalization: bool = False
    batch_norm_reestimation: bool = False
    sequential_mse: bool = False
    adaround: AdaroundConfig = Field(default_factory=AdaroundConfig)

    epochs: NonNegativeInt = 20
    optimizer: ConfigItem = Field(
        default_factory=lambda: ConfigItem(name="SGD", params={"lr": 1e-5})
    )
    scheduler: ConfigItem = Field(
        default_factory=lambda: ConfigItem(
            name="StepLR", params={"step_size": 5, "gamma": 0.1}
        )
    )

    @field_validator("config", mode="before")
    @classmethod
    def validate_config(cls, value: ParamValue) -> Any:
        if isinstance(value, str):
            try:
                fs = LuxonisFileSystem(value)
                return json.loads(fs.read_text(""))
            except Exception as e:
                raise ValueError(
                    f"Failed to load AIMET config from file '{value}': {e}"
                ) from e
        return value

    @field_serializer("default_data_type", "quant_scheme")
    def serialize_enums(self, value: Any) -> str:
        if isinstance(value, Enum):
            return value.name
        return value


class ExportConfig(ArchiveConfig):
    """How the trained model is exported and converted.

    Attributes:
        name: The name of the exported model. Left out, the model name
            is used.
        input_shape: The input shape to export with. Left out, it comes
            from the dataset.
        quantization_mode: The precision the conversion targets.
            ``FP16`` and ``FP32`` are short for the matching standard
            modes.
        reverse_input_channels: Swap the channel order in the exported
            model. It applies to the ``.blob`` export.
        scale_values: The scale of the input normalization. Left out, it
            comes from the augmentations.
        mean_values: The mean of the input normalization. Left out, it
            comes from the augmentations.
        onnx: The options of the ONNX export.
        blobconverter: Conversion to ``.blob``, which is deprecated.
        hubai: Conversion through the HubAI SDK.
        aimet: Quantization with AIMET.

    """

    name: str | None = None
    input_shape: list[int] | None = None
    quantization_mode: Annotated[
        str,
        BeforeValidator(_validate_quantization_mode),
        Field(validation_alias=AliasChoices("quantization_mode", "data_type")),
    ] = "INT8_STANDARD"
    reverse_input_channels: bool | None = None
    scale_values: list[float] | None = None
    mean_values: list[float] | None = None
    onnx: OnnxExportConfig = Field(default_factory=OnnxExportConfig)
    blobconverter: BlobconverterExportConfig = Field(
        default_factory=BlobconverterExportConfig
    )
    hubai: HubAIExportConfig = Field(default_factory=HubAIExportConfig)
    aimet: AIMETConfig = Field(default_factory=AIMETConfig)

    @field_validator("scale_values", "mean_values", mode="before")
    @classmethod
    def check_values(cls, values: ParamValue) -> Any:
        if isinstance(values, float | int):
            return [values] * 3
        return values


class StorageConfig(BaseModelExtraForbid):
    """Where Optuna keeps the study.

    Optuna reaches the database through SQLAlchemy, so every backend of
    `SQLAlchemy
    <https://docs.sqlalchemy.org/en/latest/core/engines.html#database-urls>`_
    works. With the ``postgres`` backend, a field left out is read from
    ``POSTGRES_USER``, ``POSTGRES_PASSWORD``, ``POSTGRES_HOST``,
    ``POSTGRES_PORT``, and ``POSTGRES_DB``.

    Attributes:
        active: Keep the study, so it survives the process.
        backend: The database backend.
        username: The user of the database.
        password: The password of the database.
        host: The host of the database.
        port: The port of the database.
        database: The name of the database.

    """

    active: bool = True
    backend: str = "sqlite"
    username: str | None = None
    password: SecretStr | None = None
    host: str | None = None
    port: PositiveInt | None = None
    database: str | None = None


class TunerConfig(BaseModelExtraForbid):
    """The hyperparameter search that ``luxonis_train tune`` runs.

    The tuner removes the callbacks that would upload or export a model
    of a single trial: ``UploadCheckpoint``, ``ExportOnTrainEnd``,
    ``ArchiveOnTrainEnd``, ``ConvertOnTrainEnd``, and
    ``TestOnTrainEnd``.

    Attributes:
        study_name: The name of the study.
        continue_existing_study: Add the trials to a study of the same
            name, instead of starting over.
        use_pruner: Stop a trial early once its result falls behind the
            median.
        n_trials: How many trials each process runs. Left out, it runs
            until the timeout.
        timeout: How many seconds the study runs for.
        storage: Where the study is kept.
        params: The parameters to search, and the range of each.

            A key is the path of a config field with a type suffix, as
            in ``trainer.optimizer.params.lr_float``. The suffix is one
            of ``categorical``, ``float``, ``int``, ``loguniform``,
            ``uniform``, or ``subset``, and it selects the `Optuna
            <https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html>`_
            method that samples the value.

            ``subset`` only applies to the augmentations. Give a list of
            augmentation names and a count, and each trial activates that
            many of them.
        monitor: Whether a trial is judged on the validation loss or on
            the main metric.

    """

    study_name: str = "test-study"
    continue_existing_study: bool = True
    use_pruner: bool = True
    n_trials: PositiveInt | None = 15
    timeout: PositiveInt | None = None
    storage: StorageConfig = Field(default_factory=StorageConfig)
    params: dict[str, list[str | int | float | bool | list]] = {}
    monitor: Literal["metric", "loss"] = "loss"


class Config(LuxonisConfig):
    """The root of a configuration file.

    Only the ``model`` section is required. Every other section falls
    back to its defaults, which ``luxonis_train config`` prints in full.

    Attributes:
        rich_logging: Render the logs and the progress bar as rich
            tables.
        model: The model graph. This is the one section you must write.
        loader: Where the data comes from.
        tracker: Where the metrics and the artifacts go.
        trainer: How the model trains.
        exporter: How the trained model is exported and converted.
        archiver: How the NN Archive is named and uploaded.
        tuner: The hyperparameter search, used by ``luxonis_train
            tune``.
        version: The schema version the file was written for.
            ``luxonis_train upgrade`` reads it to migrate an older file.
        ENVIRON: Environment variables, for testing only.

            A secret written here can leak into a log or a tracked run.
            Set a real environment variable, or use a ``.env`` file.

    """

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

        with suppress(ImportError):
            from luxonis_train.utils import setup_logging

            setup_logging(use_rich=use_rich)

        return data

    @classmethod
    def get_config(
        cls,
        cfg: PathType | Params | None = None,
        overrides: Params | list[str] | tuple[str, ...] | None = None,
    ) -> "Config":
        orig_cfg = cfg
        if isinstance(cfg, PathType):
            cache = Path(".cache/luxonis_train/")
            cache.mkdir(parents=True, exist_ok=True)
            cfg = LuxonisFileSystem.download(str(cfg), cache)

        if cfg is not None:
            cfg = upgrade_config(cfg)

        instance = super().get_config(cfg, overrides)
        if isinstance(orig_cfg, str):
            fs = LuxonisFileSystem(orig_cfg)
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
        warnings.
        """
        self._populate_mosaic_sizes()
        self._limit_validation_batches_for_shared_views()
        self._configure_predefined_model_defaults()
        self._add_default_callbacks()
        return self

    def _populate_mosaic_sizes(self) -> None:
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

    def _limit_validation_batches_for_shared_views(self) -> None:
        shared_views = (
            self.loader.train_view
            == self.loader.val_view
            == self.loader.test_view
        )
        if not shared_views:
            return
        if self.trainer.n_validation_batches is None:
            self.trainer.n_validation_batches = 10
            logger.warning(
                "Train, validation, and test views are the same. "
                "Automatically setting `n_validation_batches` to 10 "
                "to prevent validation/testing on the full train set. "
                "If this behavior is not desired, set "
                "`smart_cfg_auto_populate` to `False`."
            )
            return
        logger.warning(
            "Train, validation, and test views are the same. "
            "Make sure this is intended."
        )

    def _configure_predefined_model_defaults(self) -> None:
        from luxonis_train.config.predefined_versions import family_name

        predefined_model_cfg = self.model.predefined_model
        if predefined_model_cfg is None:
            return
        logger.info(
            "Predefined model detected. Adjusting parameters for best training "
            "results. If this behavior is not desired, set "
            "`smart_cfg_auto_populate` to `False`."
        )
        # `name` may carry an explicit `:vN`/`:latest` suffix; the
        # rules below apply per family, not per pinned version.
        model_name = family_name(predefined_model_cfg.name)
        if self.trainer.accumulate_grad_batches is not None:
            accumulate_grad_batches = self.trainer.accumulate_grad_batches
            logger.info(
                f"Keeping the explicitly configured "
                f"'accumulate_grad_batches' of {accumulate_grad_batches}."
            )
        else:
            accumulate_grad_batches = max(1, 64 // self.trainer.batch_size)
            self.trainer.accumulate_grad_batches = accumulate_grad_batches
            logger.info(
                f"Setting 'accumulate_grad_batches' to "
                f"{accumulate_grad_batches} "
                f"(trainer.batch_size={self.trainer.batch_size})"
            )
        loss_params = predefined_model_cfg.params.get("loss_params", {})
        if not isinstance(loss_params, dict):
            raise ValueError(  # noqa: TRY004
                f"Invalid value for loss_params: {loss_params}. Expected a dictionary."
            )
        schedule = self._update_predefined_model_loss_params(
            model_name, loss_params, accumulate_grad_batches
        )
        predefined_model_cfg.params["loss_params"] = loss_params
        if schedule is not None:
            self._set_gradient_accumulation_schedule(schedule)

    @staticmethod
    def _update_predefined_model_loss_params(
        model_name: str,
        loss_params: Params,
        accumulate_grad_batches: int,
    ) -> dict[int, int] | None:
        weights = {
            "InstanceSegmentationModel": {
                "bbox_loss_weight": 7.5,
                "class_loss_weight": 0.5,
                "dfl_loss_weight": 1.5,
            },
            "KeypointDetectionModel": {
                "iou_loss_weight": 7.5,
                "class_loss_weight": 0.5,
                "regr_kpts_loss_weight": 12,
                "vis_kpts_loss_weight": 1,
            },
            "DetectionModel": {
                "iou_loss_weight": 2.5,
                "class_loss_weight": 1,
            },
        }
        model_weights = weights.get(model_name)
        if model_weights is None:
            return None
        loss_params.update(
            {
                name: weight * accumulate_grad_batches
                for name, weight in model_weights.items()
            }
        )
        logger.info(f"{model_name}: Updated loss_params: {loss_params}")
        if model_name == "DetectionModel":
            return None
        schedule = {
            0: 1,
            1: (1 + accumulate_grad_batches) // 2,
            2: accumulate_grad_batches,
        }
        logger.info(
            f"{model_name}: Set gradient accumulation schedule to: {schedule}"
        )
        return schedule

    def _set_gradient_accumulation_schedule(
        self, schedule: dict[int, int]
    ) -> None:
        callback = next(
            (
                callback
                for callback in self.trainer.callbacks
                if callback.name == "GradientAccumulationScheduler"
            ),
            None,
        )
        if callback is None:
            return
        callback.params["scheduling"] = schedule  # type: ignore
        logger.info(
            f"GradientAccumulationScheduler callback updated with scheduling: {schedule}"
        )

    def _add_default_callbacks(self) -> None:
        default_callbacks = [
            "UploadCheckpoint",
            "TestOnTrainEnd",
            "ConvertOnTrainEnd",
        ]

        for cb_name in default_callbacks:
            if not any(cb.name == cb_name for cb in self.trainer.callbacks):
                self.trainer.callbacks.append(CallbackConfig(name=cb_name))
                logger.info(f"Added {cb_name} callback.")
