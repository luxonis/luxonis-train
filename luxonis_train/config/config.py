"""The schema of the configuration file.

Every section of the YAML file is a model here, and `Config` is the
root. The models reject an unknown key, so a typo fails at load time.

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
    """The height and the width of an image, in pixels.

    A list in the YAML file maps to it in the same order, height first.

    Attributes:
        height (int): The height of the image.
        width (int): The width of the image.

    Example:
        >>> size = ImageSize(256, 320)
        >>> size.height, size.width
        (256, 320)

    """

    height: int
    width: int


class AttachedModuleConfig(ConfigItem):
    """The fields every attached module shares.

    ``name`` is the class name of a registered loss, metric, or
    visualizer, and ``params`` reaches its constructor.

    Attributes:
        alias (str | None): The name of this module in the logs and in
            the metric keys. Without it, the identifier is the class
            name. When that name repeats one in the same node,
            `ModelConfig.check_unique_names` derives an alias.

    """

    alias: str | None = None

    @property
    def identifier(self) -> str:
        """The alias of the module, or its class name without an alias.

        Example:
            >>> MetricModuleConfig(name="F1Score").identifier
            'F1Score'
            >>> MetricModuleConfig(name="F1Score", alias="f1").identifier
            'f1'

        """
        return self.alias or self.name


class LossModuleConfig(AttachedModuleConfig):
    """A loss attached to a node.

    ``name`` is the class name of a registered loss and ``params``
    reaches its constructor. See
    `luxonis_train.attached_modules.losses`. The training step raises
    ``ValueError`` when no node produces a loss.

    Attributes:
        weight (``NonNegativeFloat``): The factor this loss contributes
            to the total. With ``0``, the loss still runs, its main
            value logs as ``0``, and it adds nothing to the total. The
            sub-losses log unscaled.

    """

    weight: NonNegativeFloat = 1.0

    @model_validator(mode="after")
    def validate_weight(self) -> Self:
        """Log a warning when ``weight`` is ``0``.

        Returns:
            ``Self``: This instance, unchanged.

        """
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
        is_main_metric (bool): Make this metric the main metric. The
            main metric selects the best checkpoint. A
            ``ReduceLROnPlateau`` scheduler in ``max`` mode monitors
            it. The tuner judges a trial on it when ``tuner.monitor``
            is ``"metric"``. At most one metric in a config can set
            it. When none does, `ModelConfig.check_main_metric` picks
            the first metric.

    """

    is_main_metric: bool = False


class FreezingConfig(BaseModelExtraForbid):
    """Whether a node trains, and when it starts.

    Attributes:
        active (bool): Freeze the node. Its weights do not update, and
            its batch normalization layers stop tracking statistics.
            The other two fields have no effect without it.
        unfreeze_after (``NonNegativeInt | NonNegativeFloat | None``):
            When to unfreeze. An integer is an epoch number. A float
            is a share of ``trainer.epochs``, truncated to a whole
            epoch. Left out, the node stays frozen for the whole run.
        lr_after_unfreeze (``NonNegativeFloat | None``): The base
            learning rate of the parameter groups of this node from the
            unfreeze epoch on.

            On that epoch it replaces the ``lr`` and the ``initial_lr``
            of each group, and the matching ``base_lrs`` entry of the
            scheduler. The scheduler then continues from its current
            position with the new base. Left out, the groups keep the
            rate the scheduler has reached.

    """

    active: bool = False
    unfreeze_after: NonNegativeInt | NonNegativeFloat | None = None
    lr_after_unfreeze: NonNegativeFloat | None = None


class ParameterPattern(BaseModelExtraForbid):
    """A pattern that selects the parameters of a node.

    Both fields are regular expressions. ``re.search`` matches them
    without anchors and without case. ``module_type: Linear`` therefore
    also claims ``LazyLinear`` and the
    ``NonDynamicallyQuantizableLinear`` inside
    ``nn.MultiheadAttention``, and ``name: conv1`` also claims
    ``branch1.conv10.weight``. Anchor the pattern, as in
    ``module_type: ^Linear$``, for an exact match.

    Attributes:
        name (str | None): A pattern matched against the dotted name of
            a parameter, relative to the node, such as
            ``stem.conv.weight``.
        module_type (str | None): A pattern matched against the class
            name of the module that owns the parameter.

    """

    name: str | None = None
    module_type: str | None = None

    @model_validator(mode="after")
    def validate_pattern(self) -> Self:
        """Require at least one field, and reject an empty string.

        Returns:
            ``Self``: This instance, unchanged.

        Raises:
            ValueError: When both fields are ``None``, or when a field
                is an empty string.

        """
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
        """Check whether a parameter matches this pattern.

        Args:
            module_type (str): The class name of the module that owns
                the parameter.
            parameter_name (str): The dotted name of the parameter.

        Returns:
            bool: ``True`` when every set field matches. A field left
            as ``None`` matches everything.

        Example:
            >>> exact = ParameterPattern(module_type="^Linear$")
            >>> exact.matches("Linear", "head.fc.weight")
            True
            >>> exact.matches("LazyLinear", "head.fc.weight")
            False
            >>> loose = ParameterPattern(name="conv1")
            >>> loose.matches("Conv2d", "conv10.bias")
            True

        """
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
    constructor. See `luxonis_train.schedulers`.

    ``CosineAnnealingLR`` without ``T_max`` gets ``trainer.epochs`` when
    the trainer builds the schedulers, with a warning.

    """

    name: str = "ConstantLR"

    def get_sequential_lr_params(self) -> "SequentialLRParams":
        """Parse ``params`` as the arguments of ``SequentialLR``.

        Returns:
            SequentialLRParams: The child schedulers, the milestones,
            and the last epoch.

        Raises:
            RuntimeError: When ``name`` is not ``"SequentialLR"``.
            ValueError: When ``params`` lacks ``schedulers`` or
                ``milestones``.

        Example:
            >>> cfg = SchedulerConfig(
            ...     name="SequentialLR",
            ...     params={
            ...         "schedulers": [
            ...             {"name": "LinearLR"},
            ...             {"name": "ConstantLR"},
            ...         ],
            ...         "milestones": [5],
            ...     },
            ... )
            >>> seq = cfg.get_sequential_lr_params()
            >>> [s.name for s in seq.schedulers]
            ['LinearLR', 'ConstantLR']
            >>> seq.milestones, seq.last_epoch
            ([5], -1)

        """
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
        """Copy this config into a `FinetuningSchedulerConfig`.

        Returns:
            FinetuningSchedulerConfig: An override with the same
            ``name`` and ``params``.

        Example:
            >>> cfg = SchedulerConfig(name="StepLR", params={"step_size": 5})
            >>> cfg.to_finetuning()
            FinetuningSchedulerConfig(name='StepLR', params={'step_size': 5})

        """
        return FinetuningSchedulerConfig(name=self.name, params=self.params)


class SequentialLRParams(BaseModelExtraForbid):
    """The parameters ``SequentialLR`` requires.

    `SchedulerConfig.get_sequential_lr_params` builds it from the
    ``params`` of a ``SequentialLR`` config.

    Attributes:
        schedulers (list[SchedulerConfig]): The child schedulers, in the
            order they run.
        milestones (list[int]): The epochs at which one child hands
            over to the next.
        last_epoch (int): The epoch to resume from. ``-1`` starts from
            the beginning.

    """

    schedulers: list[SchedulerConfig]
    milestones: list[int]
    last_epoch: int = -1


class FinetuningSchedulerConfig(SchedulerConfig):
    """A scheduler override for a finetuning entry.

    An override that omits ``name``, or that repeats the name of the
    base scheduler, keeps that name and merges its ``params`` into the
    base ``params``. An override that names a different scheduler drops
    the base ``params`` and uses only its own. The base is
    ``trainer.scheduler``, or the scheduler a training strategy
    provides.

    Attributes:
        name (str | None): The class name of the scheduler, or ``None``
            to keep the base one.

    """

    name: str | None = None


class OptimizerConfig(ConfigItem):
    """The optimizer to use.

    ``name`` is the class name of any optimizer of ``torch.optim``, and
    ``params`` reaches its constructor. See `luxonis_train.optimizers`.

    """

    name: str = "Adam"

    def to_finetuning(self) -> "FinetuningOptimizerConfig":
        """Copy this config into a `FinetuningOptimizerConfig`.

        Returns:
            FinetuningOptimizerConfig: An override with the same
            ``name`` and ``params``.

        Example:
            >>> cfg = OptimizerConfig(name="SGD", params={"lr": 0.01})
            >>> cfg.to_finetuning()
            FinetuningOptimizerConfig(name='SGD', params={'lr': 0.01})

        """
        return FinetuningOptimizerConfig(name=self.name, params=self.params)


class FinetuningOptimizerConfig(OptimizerConfig):
    """An optimizer override for a finetuning entry.

    An override that omits ``name``, or that repeats the name of the
    base optimizer, keeps that name and merges its ``params`` into the
    base ``params``. An override that names a different optimizer drops
    the base ``params`` and uses only its own. The base is
    ``trainer.optimizer``, or the optimizer a training strategy
    provides.

    Attributes:
        name (str | None): The class name of the optimizer, or ``None``
            to keep the base one.

    """

    name: str | None = None


class FinetuningConfig(BaseModelExtraForbid):
    """A different optimizer or scheduler for part of a node.

    Use it to train a pretrained backbone at a lower learning rate than
    a fresh head.

    The trainer evaluates the entries of a node in order, and the first
    entry whose pattern matches a parameter claims it. Put the specific
    rules first and the general ones last. An entry that claims no
    parameter raises ``ValueError`` when the optimizers are built.

    Entries that share an optimizer name, a scheduler name, and the
    scheduler ``params`` collapse into one optimizer. Each entry keeps
    its own parameter group, so a different ``lr`` between entries
    still applies. Any other difference produces a separate inner optimizer,
    and one `CompositeOptimizer
    <luxonis_train.optimizers.composite_optimizer.CompositeOptimizer>`
    drives them all. Training therefore stays in the automatic
    optimization of Lightning, and gradient accumulation and gradient
    clipping keep working.

    Every parameter reaches an optimizer. A parameter that no entry
    claims falls to a default group that uses the base optimizer and
    scheduler. A frozen parameter stays in its group, and the optimizer
    skips it while it is frozen. A node that unfreezes mid-training
    therefore already has an optimizer.

    Every parameter group carries a name, so
    `lightning.pytorch.callbacks.LearningRateMonitor` logs one series
    for each group instead of ``pg1``, ``pg2``, and so on. A finetuning
    entry is named ``<node>/<index>``, a strategy group
    ``strategy/<tag>``, and the default group ``default``. A group that
    is scoped to one node gains the node name, as in ``default/<node>``.
    A configuration that produces a single group leaves it unnamed.

    Attributes:
        parameters (list[ParameterPattern] | None): The parameters this
            entry claims. Left out, the entry claims every parameter of
            the node. A plain string is short for a pattern on the
            parameter name.
        optimizer (FinetuningOptimizerConfig | None): The optimizer for
            the claimed parameters. Left out, the base optimizer
            applies.
        scheduler (FinetuningSchedulerConfig | None): The scheduler for
            the claimed parameters. Left out, the base scheduler
            applies.

    """

    parameters: list[ParameterPattern] | None = None
    optimizer: FinetuningOptimizerConfig | None = None
    scheduler: FinetuningSchedulerConfig | None = None

    @field_validator("parameters", mode="before")
    @classmethod
    def validate_parameters(cls, value: Any) -> Any:
        """Normalize ``parameters`` into a list of `ParameterPattern`.

        A single string, dictionary, or pattern becomes a one-element
        list. Inside the list, a string becomes a pattern on the
        parameter name, and a dictionary becomes a pattern with the
        given fields. Any other value passes through, so pydantic
        reports it.

        Args:
            value (``Any``): The raw value of the ``parameters`` field.

        Returns:
            ``Any``: The list of patterns, or ``value`` unchanged when
            it is not a string, a dictionary, a pattern, or a list.

        Raises:
            ValueError: When the list is empty. Pydantic reports it as
                a validation error.
            TypeError: When a list item is not a string, a dictionary,
                or a `ParameterPattern`.

        Example:
            >>> FinetuningConfig(parameters="backbone").parameters
            [ParameterPattern(name='backbone', module_type=None)]
            >>> cfg = FinetuningConfig(parameters=[{"module_type": "^Conv"}])
            >>> cfg.parameters[0].module_type
            '^Conv'

        """
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
        alias (str | None): The name of this node in the graph. Other
            nodes refer to it in ``inputs``, the checkpoint keys carry
            it, and the logs use it. Without it, the identifier is the
            class name.
        inputs (list[str]): The nodes that feed this one. A node with
            neither ``inputs`` nor ``input_sources`` reads every output
            of the loader. From the second node on,
            `ModelConfig.validate_nodes` fills in an omitted
            ``inputs``.
        input_sources (list[str]): The loader outputs that feed this
            node directly, by name.
        remove_on_export (bool): Skip this node in the exported model.
        task_name (str | None): The dataset task this node reads.
            Without it, the node takes the single task of the dataset.
            A head on a dataset with several tasks must set it.
        metadata_task_override (str | dict[str, str] | None): New names
            for the metadata labels the task of the node requires. A
            string renames the single required label. A dictionary
            maps the default label names to the new ones.
        variant (str | None): The variant of the node. ``"default"``
            selects the default variant of the node class. ``"none"``
            or ``None`` builds the node without variant parameters.
            Each node docstring lists the variants it declares.
        losses (list[LossModuleConfig]): The losses attached to this
            node.
        metrics (list[MetricModuleConfig]): The metrics attached to
            this node.
        visualizers (list[AttachedModuleConfig]): The visualizers
            attached to this node.
        finetuning (list[FinetuningConfig]): The optimizer and
            scheduler overrides for this node. A single entry does not
            need the list.
        freezing (FreezingConfig): Whether this node trains, and when
            it starts.

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
        """Wrap a single ``finetuning`` mapping into a list.

        Args:
            value (``Any``): The raw value of the ``finetuning`` field.

        Returns:
            ``Any``: A one-element list when ``value`` is a dictionary,
            otherwise ``value`` unchanged.

        Example:
            >>> node = NodeConfig(name="ClassificationHead", finetuning={})
            >>> node.finetuning
            [FinetuningConfig(parameters=None, optimizer=None, scheduler=None)]

        """
        if isinstance(value, dict):
            return [value]
        return value

    @property
    def identifier(self) -> str:
        """The alias of the node, or its class name without an alias.

        Example:
            >>> NodeConfig(name="EfficientRep", alias="backbone").identifier
            'backbone'
            >>> NodeConfig(name="EfficientRep").identifier
            'EfficientRep'

        """
        return self.alias or self.name


class PredefinedModelConfig(ConfigItem):
    """A reference to a predefined model, with its parameters.

    ``name`` selects the model family, such as ``DetectionModel``, and
    ``params`` reaches its constructor. The name may carry a version
    suffix, as in ``DetectionModel:v1`` or ``DetectionModel:latest``.
    See `luxonis_train.config.predefined_models` for the models and
    the parameters each one accepts.

    Attributes:
        variant (str | None): The variant to build. ``"default"``
            selects the default variant of the model, and each model
            documents the others. ``"none"`` or ``None`` builds the
            model without variant parameters. A non-empty ``variant``
            inside ``params`` takes precedence.
        version (``int | Literal["latest"]``): The version of the
            model. ``"latest"`` follows the newest one, and an integer
            pins the graph a config was written against. A version in
            ``name`` must agree with ``version``, unless ``version`` is
            ``"latest"``.
        include_losses (bool): Add the losses of the model.
        include_metrics (bool): Add the metrics of the model.
        include_visualizers (bool): Add the visualizers of the model.

    """

    variant: str | Literal["default", "none"] | None = "default"
    version: int | Literal["latest"] = "latest"
    include_losses: bool = True
    include_metrics: bool = True
    include_visualizers: bool = True


class ModelConfig(BaseModelExtraForbid):
    """The model graph, or the predefined model that generates one.

    Build the graph by hand with ``nodes``, or name a predefined model
    and let it contribute the nodes. A config may hold both: the nodes
    a predefined model generates follow the listed ones.

    Attributes:
        name (str): The name of the model. It names the checkpoint
            files, the exported files, and the model on HubAI.
        predefined_model (PredefinedModelConfig | None): A predefined
            model that generates the nodes. ``model_dump`` leaves it
            out, because the generated nodes are dumped in ``nodes``.
        weights (``FilePath | None``): An existing local checkpoint to
            start from. ``trainer.resume_training`` decides whether the
            optimizer state comes with it. ``model_dump`` leaves it
            out.
        nodes (list[NodeConfig]): The nodes of the graph.
        outputs (list[str]): The identifiers of the nodes whose outputs
            the model returns. Left empty, `check_graph` fills it with
            the nodes that feed no other node.

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
        """Fill in the ``inputs`` a node leaves out.

        The rule assumes a linear topology with several heads:
        ``backbone -> (neck) -> head1, head2, ...``. A node with
        neither ``inputs`` nor ``input_sources`` gets one input. Before
        the first head, that input is the node listed just above. From
        the first head on, it is the last node before that head, so
        every head reads from the same body. A node counts as a head
        when its ``name`` contains ``Head``. The first node keeps its
        empty ``inputs``, so it reads from the loader. The validator
        logs a warning for each filled input.

        Args:
            nodes (``ParamValue``): The raw value of the ``nodes``
                field.

        Returns:
            ``Any``: The same list with ``inputs`` filled in, or
            ``nodes`` unchanged when it is not a list of dictionaries.

        Raises:
            ValueError: When a node has no ``name``.

        """
        if not check_type(nodes, list[dict]):
            return nodes

        return cls._populate_implicit_node_inputs(nodes)

    @staticmethod
    def _populate_implicit_node_inputs(
        nodes: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Apply the ``inputs`` rule of `validate_nodes` in place."""
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
        """Append the nodes a predefined model generates.

        It resolves the model class from ``predefined_model.name`` and
        ``predefined_model.version``. It logs the name, and the registry
        key it resolved to when that differs. It builds the model from
        ``params`` and ``variant``, and appends the nodes it generates
        to ``nodes``. The ``include_*`` flags decide whether the losses,
        the metrics, and the visualizers come along. Without
        ``predefined_model``, nothing changes.

        Returns:
            ``Self``: This instance, with the generated nodes appended.

        Raises:
            ValueError: When the family or the version is unknown, or
                when the version in ``name`` conflicts with
                ``version``.

        """
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
        """Ensure that at most one metric is the main metric.

        When no metric sets ``is_main_metric``, the first metric of the
        first node with metrics becomes the main metric, with a
        warning. A config without metrics only draws a warning. The
        main metric is logged.

        Returns:
            ``Self``: This instance, with one main metric when any
            metric exists.

        Raises:
            ValueError: When more than one metric sets
                ``is_main_metric``.

        """
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
        """Reject a cyclic graph and fill in ``outputs``.

        Left empty, ``outputs`` becomes the identifiers of the nodes
        that no other node lists in ``inputs``, in the order of
        ``nodes``.

        Returns:
            ``Self``: This instance, with ``outputs`` filled in.

        Raises:
            ValueError: When the graph has a cycle, or when ``nodes``
                is not empty and no output remains.

        """
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
        """Reject a ``/`` in the name or the alias of a node or module.

        ``/`` separates the parts of the logged keys, so a name may not
        contain it.

        Returns:
            ``Self``: This instance, unchanged.

        Raises:
            ValueError: When a node, a loss, a metric, or a visualizer
                has a ``/`` in its ``name`` or ``alias``.

        """
        for node in self.nodes:
            for module in self._node_modules(node):
                self._validate_module_characters(module)

        return self

    @model_validator(mode="after")
    def check_unique_names(self) -> Self:
        """Give every attached module of a node a unique identifier.

        The check treats the losses, the metrics, and the visualizers
        of a node as three groups, each together with the node itself.
        A module without an alias whose class name repeats an earlier
        identifier gets the alias ``<name>_<alias of the node>``. That
        alias reads ``<name>_None`` when the node has no alias. When an
        explicit alias, or that derived alias, repeats an earlier
        identifier, the check appends a counter, as in ``<alias>_0``,
        and logs a warning.

        Returns:
            ``Self``: This instance, with unique aliases.

        """
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
        """Rename duplicates among ``node`` and ``modules`` in place."""
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
                original_alias = module.alias
                while (new_alias := f"{original_alias}_{node_index}") in names:
                    node_index += 1
                logger.warning(
                    f"Duplicate name: {module.alias}. Renaming to {new_alias}."
                )
                module.alias = new_alias
                node_index += 1

            names.add(module.alias or module.name)

    @property
    def head_nodes(self) -> list[NodeConfig]:
        """The nodes whose registered class is a head.

        A node counts when its ``name`` is registered and the class
        subclasses `BaseHead`. An unregistered name is left out.

        """
        from luxonis_train.nodes import BaseHead

        return [
            node
            for node in self.nodes
            if issubclass(NODES._module_dict.get(node.name, object), BaseHead)
        ]


class TrackerConfig(BaseModelExtraForbid):
    """Where the metrics, the images, and the checkpoints go.

    More than one backend can be active at once, and at least one must
    be. Weights and Biases and MLFlow need ``project_name`` or
    ``project_id``.

    Attributes:
        project_name (str | None): The project the run belongs to.
        project_id (str | None): The project identifier. MLFlow uses it
            instead of ``project_name`` when both are set. Weights and
            Biases then uses ``project_name``.
        run_name (str | None): The name of the run. Left out, the
            tracker generates one.
        run_id (str | None): An existing MLFlow run to continue.
        save_directory (pathlib.Path): The directory that holds one
            subdirectory for each run, with the logs, the checkpoints,
            and the exported files. ``model_dump`` leaves it out.
        is_tensorboard (bool): Log to TensorBoard.
        is_wandb (bool): Log to Weights and Biases.
        wandb_entity (str | None): The Weights and Biases entity that
            owns the run. Required when ``is_wandb`` is set.
        is_mlflow (bool): Log to MLFlow. It needs the
            ``MLFLOW_TRACKING_URI`` environment variable.

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
        name (str): The class name of a registered loader.
        image_source (str): The name of the loader output that holds
            the main image.
        train_view (list[str]): The dataset splits to train on. A
            string names a single split.
        val_view (list[str]): The dataset splits to validate on.
        test_view (list[str]): The dataset splits to test on.

    """

    name: str = "LuxonisLoaderTorch"
    image_source: str = "image"
    train_view: list[str] = ["train"]
    val_view: list[str] = ["val"]
    test_view: list[str] = ["test"]

    @field_serializer("params")
    def serialize_params(self, info: SerializationInfo) -> Any:
        """Dump ``params`` without the fields of a ``DummyLoader``.

        `LuxonisModel` falls back to a `DummyLoader` when
        ``allow_empty_dataset`` is set and the loader fails to build.
        Its ``n_classes``, ``n_keypoints``, and ``class_names`` describe
        the missing dataset, so the dump drops them and the saved config
        loads with a real loader again.

        Args:
            info (``SerializationInfo``): The serialization context of
                pydantic. Unused.

        Returns:
            ``Params``: A copy of ``params``, without the three keys
            when ``name`` is ``"DummyLoader"``.

        """
        data = self.params.copy()
        if self.name == "DummyLoader":
            data.pop("n_classes", None)
            data.pop("n_keypoints", None)
            data.pop("class_names", None)
        return data

    @field_serializer("name")
    def serialize_name(self, info: SerializationInfo) -> str:
        """Dump ``name`` without the ``DummyLoader`` fallback.

        Args:
            info (``SerializationInfo``): The serialization context of
                pydantic. Unused.

        Returns:
            str: ``"LuxonisLoaderTorch"`` when ``name`` is
            ``"DummyLoader"``, otherwise ``name``.

        Example:
            >>> LoaderConfig(name="DummyLoader").model_dump()["name"]
            'LuxonisLoaderTorch'

        """
        if self.name == "DummyLoader":
            return "LuxonisLoaderTorch"
        return self.name

    @field_validator("train_view", "val_view", "test_view", mode="before")
    @classmethod
    def validate_view(cls, splits: ParamValue) -> list[Any]:
        """Wrap a single split name into a list.

        Args:
            splits (``ParamValue``): The raw value of ``train_view``,
                ``val_view``, or ``test_view``.

        Returns:
            ``list[Any]``: A one-element list for a string, otherwise
            the list unchanged.

        Raises:
            TypeError: When the value is neither a string nor a list.

        Example:
            >>> LoaderConfig(train_view="train").train_view
            ['train']
            >>> LoaderConfig(val_view=["val", "test"]).val_view
            ['val', 'test']

        """
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
        """Normalize ``dataset_type`` in ``params`` to lower case.

        It matches the value against the members of
        ``luxonis_ml.enums.DatasetType`` without case, and stores it in
        lower case. Without ``dataset_type``, ``params`` stays as it
        is.

        Returns:
            ``Self``: This instance, with ``dataset_type`` normalized.

        Raises:
            TypeError: When ``dataset_type`` is not a string.
            ValueError: When ``dataset_type`` names no known dataset
                type.

        Example:
            >>> LoaderConfig(params={"dataset_type": "COCO"}).params
            {'dataset_type': 'coco'}

        """
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
    """The normalization of the image values.

    The default ``params`` hold the ImageNet mean and standard
    deviation. A ``Normalize`` entry in ``augmentations`` overrides
    ``params``; see `PreprocessingConfig.check_normalize`. The loader
    applies it on every stage, as the last pixel-level augmentation.
    The resize may still follow it.

    Attributes:
        active (bool): Normalize the images.
        params (``Params``): The parameters of the ``Albumentations``
            ``Normalize`` transform.

    """

    active: bool = True
    params: Params = {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    }


class AugmentationConfig(ConfigItem):
    """One augmentation.

    ``name`` is the name of an ``Albumentations`` transform, or of a
    custom transform of ``luxonis_ml`` such as ``Mosaic4``, ``MixUp``,
    or ``CutMix``, and ``params`` reaches its constructor.

    ``HorizontalFlip``, ``VerticalFlip``, and ``Transpose`` move the
    keypoints but do not swap a left keypoint with its right
    counterpart. Use ``HorizontalSymmetricKeypointsFlip``,
    ``VerticalSymmetricKeypointsFlip``, or
    ``TransposeSymmetricKeypoints`` for a symmetric skeleton.

    Attributes:
        active (bool): Apply this augmentation. An inactive entry stays
            in the config but does not reach the loader.
        use_for_resizing (bool): Resize with this augmentation instead
            of the default one.

            `PreprocessingConfig.check_use_for_resizing` overrides its
            ``height`` and ``width`` with ``train_image_size``, and
            sets ``p`` to ``1.0`` when it is left out. With ``p`` of
            ``1``, ``keep_aspect_ratio`` has no effect. With a lower
            ``p``, the default resize handles the other images, so
            every image is resized. Only one augmentation can carry
            this flag.
        apply_on_stages (``list[Literal["train", "val", "test"]]``): The
            stages that apply this augmentation. The loader applies
            ``Normalize`` on every stage, whatever this field says.

    """

    active: bool = True
    use_for_resizing: bool = False
    apply_on_stages: list[Literal["train", "val", "test"]] = Field(
        default_factory=lambda: ["train"]
    )


class PreprocessingConfig(BaseModelExtraForbid):
    """The resizing and the augmentations applied to every image.

    The augmentations come from `Albumentations
    <https://albumentations.ai/docs/>`_. Name any pixel-level or
    spatial-level transform of that library here. The batch
    augmentations ``Mosaic4``, ``MixUp``, and ``CutMix`` of
    ``luxonis_ml`` are also available.

    Attributes:
        train_image_size (ImageSize): The size every image is resized
            to, as height and width.
        keep_aspect_ratio (bool): Pad the image to the size instead of
            stretching it.
        color_space (``Literal["RGB", "BGR", "GRAY"]``): The color space
            the model trains on.
        normalize (NormalizeAugmentationConfig): The normalization
            applied to every image, on every stage.
        augmentations (list[AugmentationConfig]): The augmentations.
            The loader groups them by kind, so the order here is not
            the order they apply in.

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
        """Reconcile ``normalize`` with a ``Normalize`` augmentation.

        The check removes a ``Normalize`` entry from ``augmentations``
        and copies its ``params`` into ``normalize.params``. It logs a
        warning when ``normalize`` is active, because the entry wins.
        When ``normalize`` is active, it then appends a ``Normalize``
        augmentation with ``normalize.params``. When ``normalize`` is
        inactive, no normalization runs.

        Returns:
            ``Self``: This instance, with ``augmentations`` and
            ``normalize`` reconciled.

        Example:
            >>> [a.name for a in PreprocessingConfig().augmentations]
            ['Normalize']
            >>> cfg = PreprocessingConfig(normalize={"active": False})
            >>> cfg.augmentations
            []

        """
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
        """Align the resize augmentation with ``train_image_size``.

        For each augmentation with ``use_for_resizing``, the check sets
        ``p`` to ``1.0`` when it is left out, and sets ``height`` and
        ``width`` to ``train_image_size``. It logs a warning when the
        sizes differed, and another one when ``keep_aspect_ratio`` is
        set and ``p`` is ``1``, because the flag then has no effect.

        Returns:
            ``Self``: This instance, with the resize parameters aligned.

        Example:
            >>> cfg = PreprocessingConfig(
            ...     train_image_size=[256, 320],
            ...     keep_aspect_ratio=False,
            ...     augmentations=[
            ...         {
            ...             "name": "RandomResizedCrop",
            ...             "use_for_resizing": True,
            ...             "params": {"height": 256, "width": 320},
            ...         }
            ...     ],
            ... )
            >>> cfg.augmentations[0].params
            {'height': 256, 'width': 320, 'p': 1.0}

        """
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
        """Dump the section without the ``Normalize`` augmentation.

        `check_normalize` appends that augmentation on every load, so
        the dump drops it to keep a saved config stable.

        Args:
            info (``SerializationInfo``): The serialization context of
                pydantic. Unused.

        Returns:
            ``Params``: The public fields of the section, with
            ``Normalize`` removed from ``augmentations``.

        Example:
            >>> PreprocessingConfig().model_dump()["augmentations"]
            []

        """
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
        """Return the active augmentations as fresh configs.

        Each copy keeps ``name``, ``params``, ``use_for_resizing``, and
        ``apply_on_stages``. The loader receives this list.

        Returns:
            list[AugmentationConfig]: One copy for each augmentation
            whose ``active`` is ``True``, in the same order.

        Example:
            >>> cfg = PreprocessingConfig(
            ...     normalize={"active": False},
            ...     augmentations=[
            ...         {"name": "HorizontalFlip"},
            ...         {"name": "Rotate", "active": False},
            ...     ],
            ... )
            >>> [a.name for a in cfg.get_active_augmentations()]
            ['HorizontalFlip']

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
    callbacks, and `Config.smart_auto_populate` for the ones added by
    default.

    Attributes:
        active (bool): Run this callback. An inactive callback stays in
            the config, and the trainer logs that it skips it.

    """

    active: bool = True


class TrainerConfig(BaseModelExtraForbid):
    """Everything about how the model trains.

    The trainer calls the callbacks in the order this section lists
    them, except that ``EMACallback`` moves to the front.

    Attributes:
        preprocessing (PreprocessingConfig): The resizing and the
            augmentations.
        precision (``Literal["16-mixed", "32"]``): The numeric precision
            of the training, as Lightning defines it.
        accelerator (``Literal["auto", "cpu", "gpu", "tpu"]``): The
            hardware to train on.
        devices (int | list[int] | str): How many devices to use, or
            which ones.
        strategy (``Literal["auto", "ddp"]``): The distribution
            strategy.
        n_sanity_val_steps (int): How many validation batches to run
            before the training starts.
        profiler (``Literal["simple", "advanced"] | None``): The
            Lightning profiler, which reports where the time goes.
        matmul_precision (``Literal["medium", "high", "highest"] | None``):
            The internal precision of a float32 matrix multiplication.
        seed (int | None): The seed of every random number generator.
            Set it to make a run reproducible.
        n_validation_batches (``PositiveInt | Literal[-1] | None``): How
            many batches of the validation view and of the test view
            to evaluate.

            A positive number takes the first batches of each view,
            and ``-1`` takes the views in full. Without it, each view
            runs in full, unless `Config.smart_auto_populate` sets it.
        deterministic (``bool | Literal["warn"] | None``): Use the
            deterministic kernels of PyTorch. Some layers have none,
            and ``"warn"`` lets those through. Left out with a
            ``seed``, it becomes ``True``.
        smart_cfg_auto_populate (bool): Fill in the fields a config
            leaves out, and log what was filled. See
            `Config.smart_auto_populate` for the rules.
        batch_size (``PositiveInt``): How many samples one step uses.
        accumulate_grad_batches (``PositiveInt | None``): How many
            batches to accumulate before an optimizer step. It raises
            the effective batch size without more memory. A
            ``GradientAccumulationScheduler`` in ``callbacks`` takes
            precedence over it.
        gradient_clip_val (``NonNegativeFloat | None``): The value to
            clip the gradients at. Left out, the gradients are not
            clipped.
        gradient_clip_algorithm (``Literal["norm", "value"] | None``):
            Clip the gradients by their norm, or element by element.
        use_weighted_sampler (bool): Not implemented. ``True`` raises
            ``NotImplementedError`` when the loaders are built.
        epochs (``PositiveInt``): How many epochs to train.
        overfit_batches (``NonNegativeInt``): Train and validate on
            this many batches only.

            Use it to check that a config learns at all, or to test a
            visualizer. Lightning turns off the shuffling, so each
            stage keeps its batches across epochs. Training and
            validation draw from their own views. A warning asks for
            ``seed`` when it is left out.
        resume_training (bool): Continue the run that ``model.weights``
            came from.

            The optimizer, the scheduler, and the epoch count all
            continue. A warning fires when ``epochs`` is lower than the
            ``epochs`` the checkpoint was trained with. Left false,
            only the weights load and the training state starts fresh,
            which is what a finetuning run wants.
        strict_weights_loading (bool): Require every checkpoint key to
            match the model. Left off, the loader remaps a mismatched
            node through the execution order, or loads the keys that
            match. On a resume with this flag, only the keys through
            which a loss, a metric, or a visualizer refers to its node
            may still mismatch.
        n_workers (``NonNegativeInt``): How many worker processes load
            the data. `check_n_workers_platform` sets it to ``0`` on
            Windows and macOS.
        validation_interval (``Literal[-1] | PositiveInt``): How many
            epochs pass between two validation runs.
            `check_validation_interval` clamps a value above ``epochs``
            to ``epochs``.
        run_validation_after_first_epoch (bool): Also validate after
            the first epoch, whatever ``validation_interval`` says.
        n_log_images (``NonNegativeInt``): How many visualization
            images each node logs on a validation or test epoch.
        skip_last_batch (bool): Drop the last training batch of an
            epoch when it is smaller than the others.
        pin_memory (bool): Pin the memory of the data loaders, which
            speeds up the transfer to a GPU.
        log_sub_metrics (bool): Log the parts of a metric, such as
            ``map_small`` beside ``map``.
        log_sub_losses (bool): Log the parts of a loss beside the
            total.
        save_top_k (``Literal[-1] | NonNegativeInt``): How many
            checkpoints to keep, for the best loss and for the best
            metric each. ``-1`` keeps every one.
        callbacks (list[CallbackConfig]): The callbacks to run.
        optimizer (OptimizerConfig): The optimizer, for the parameters
            that no finetuning entry and no strategy claims. It is also
            the base a `FinetuningOptimizerConfig` merges into.
        scheduler (SchedulerConfig): The scheduler, for the same
            parameters. It is also the base a
            `FinetuningSchedulerConfig` merges into.
        training_strategy (``ConfigItem | None``): A strategy that owns
            the optimization schedule. When it provides its own base
            optimizer and scheduler, they replace ``optimizer`` and
            ``scheduler``. See `luxonis_train.strategies`.

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
        """Cast string epoch keys of an accumulation schedule to ``int``.

        YAML may write the epochs of ``params.scheduling`` of a
        ``GradientAccumulationScheduler`` callback as strings. The
        callback needs integers. A key that is not a string of digits
        stays as it is.

        Returns:
            ``Self``: This instance, with integer keys.

        Example:
            >>> cfg = TrainerConfig(
            ...     n_workers=0,
            ...     callbacks=[
            ...         {
            ...             "name": "GradientAccumulationScheduler",
            ...             "params": {"scheduling": {"0": 1, "4": 2}},
            ...         }
            ...     ],
            ... )
            >>> cfg.callbacks[0].params["scheduling"]
            {0: 1, 4: 2}

        """
        for callback in self.callbacks:
            if callback.name != "GradientAccumulationScheduler":
                continue

            scheduling = callback.params.get("scheduling")
            if not isinstance(scheduling, Mapping):
                # Leave a value that is not a mapping for the callback
                # to reject.
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
        """Set ``deterministic`` to ``True`` when ``seed`` is set.

        It applies only when ``deterministic`` is left out, and it logs
        a warning, because some layers have no deterministic kernel.

        Returns:
            ``Self``: This instance, with ``deterministic`` resolved.

        """
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
        """Warn when ``overfit_batches`` is set without ``seed``.

        Returns:
            ``Self``: This instance, unchanged.

        """
        if self.overfit_batches > 0 and self.seed is None:
            logger.warning(
                "Using `overfit_batches` without setting `seed` may cause "
                "different batches to be selected each run due to shuffling. "
                "Consider setting `trainer.seed` for reproducible results."
            )
        return self

    @model_validator(mode="after")
    def check_n_workers_platform(self) -> Self:
        """Force ``n_workers`` to ``0`` on Windows and macOS.

        It logs a warning when it changes the value.

        Returns:
            ``Self``: This instance, with ``n_workers`` adjusted.

        """
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
        """Clamp ``validation_interval`` to ``epochs``.

        A larger interval would never validate, so no checkpoint would
        be written. It logs a warning when it changes the value.

        Returns:
            ``Self``: This instance, with the interval clamped.

        """
        if self.validation_interval > self.epochs:
            logger.warning(
                "Setting `validation_interval` same as `epochs`, "
                "otherwise no checkpoint would be generated."
            )
            self.validation_interval = self.epochs
        return self

    @model_validator(mode="after")
    def reorder_callbacks(self) -> Self:
        """Move ``EMACallback`` to the front of ``callbacks``.

        The EMA weights must update before the other callbacks run.
        The sort is stable, so the other callbacks keep their order.

        Returns:
            ``Self``: This instance, with the callbacks reordered.

        Example:
            >>> cfg = TrainerConfig(
            ...     n_workers=0,
            ...     callbacks=[
            ...         {"name": "UploadCheckpoint"},
            ...         {"name": "EMACallback"},
            ...     ],
            ... )
            >>> [cb.name for cb in cfg.callbacks]
            ['EMACallback', 'UploadCheckpoint']

        """
        self.callbacks.sort(key=lambda v: 0 if v.name == "EMACallback" else 1)
        return self

    @model_validator(mode="after")
    def check_convert_callbacks(self) -> Self:
        """Resolve the overlap between the conversion callbacks.

        ``ConvertOnTrainEnd`` exports, archives, and converts. Beside an
        active one, the check deactivates an active ``ExportOnTrainEnd``
        or ``ArchiveOnTrainEnd`` and logs a warning. Without
        ``ConvertOnTrainEnd``, an active pair of ``ExportOnTrainEnd``
        and ``ArchiveOnTrainEnd`` only draws a warning that suggests
        ``ConvertOnTrainEnd``.

        Returns:
            ``Self``: This instance, with the redundant callbacks
            deactivated.

        """
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

    ``opset_version`` and ``dynamic_axes`` reach ``torch.onnx.export``.

    Attributes:
        opset_version (``PositiveInt``): The ONNX opset to target.
        dynamic_axes (``Params | None``): The axes that stay dynamic in
            the exported model, keyed by input or output name.
        disable_onnx_simplification (bool): Keep the graph as exported,
            without the ``onnxsim`` pass.
        unique_onnx_initializers (bool): Duplicate an initializer that
            several nodes share, so each node owns its own copy.

    """

    opset_version: PositiveInt = 16
    dynamic_axes: Params | None = None
    disable_onnx_simplification: bool = False
    unique_onnx_initializers: bool = False


class BlobconverterExportConfig(BaseModelExtraForbid):
    """Conversion to the ``.blob`` format.

    ``blobconverter`` is deprecated and only converts for RVC2. Use
    `HubAIExportConfig` instead. It supports only ``FP16`` and
    ``FP32``, and it falls back to ``FP16`` for any other
    ``quantization_mode``, with a warning.

    Attributes:
        active (bool): Convert to ``.blob``.
        shaves (int): How many SHAVE cores the blob targets.
        version (``Literal["2021.2", "2021.3", "2021.4", "2022.1", "2022.3_RVC3"]``):
            The OpenVINO version to convert with.

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

    The upload targets the HubAI model named ``model.name``. When that
    model does not exist, the SDK creates it. The converted variant is
    named ``<model.name>:<dataset name>``, or ``model.name`` when the
    loader has no dataset name. The `Luxonis documentation
    <https://docs.luxonis.com/cloud/hubai/model-registry/concepts/>`_
    describes models, variants, and versions.

    Attributes:
        active (bool): Convert through the HubAI SDK.
        platform (``Literal["rvc2", "rvc3", "rvc4", "hailo"] | None``):
            The device to convert for. It is required when ``active``
            is true. ``"hailo"`` is not supported yet.
        params (``Params``): Extra keyword arguments for the conversion
            call of the SDK.
        delete_remote_model (bool): Clean up on HubAI when the
            conversion ends. Delete the model this run created, also
            after a failure. When the model existed before, delete only
            the new variant, and only after a success.

    """

    active: bool = False
    platform: Literal["rvc2", "rvc3", "rvc4", "hailo"] | None = None
    params: Params = Field(default_factory=dict)
    delete_remote_model: bool = False

    @model_validator(mode="after")
    def validate_platform(self) -> Self:
        """Require ``platform`` when active, and reject Hailo.

        Returns:
            ``Self``: This instance, unchanged.

        Raises:
            ValueError: When ``active`` is true and ``platform`` is
                ``None``.
            NotImplementedError: When ``platform`` is ``"hailo"``.

        """
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
        name (str | None): The name of the archive, without the suffix.
            Without it, ``model.name`` names the archive.
        upload_to_run (bool): Attach the archive to the tracked run.
        upload_url (str | None): A remote location to upload the
            archive to as well.

    """

    name: str | None = None
    upload_to_run: bool = True
    upload_url: str | None = None


def _validate_quantization_mode(value: str) -> str:
    """Upper-case the mode, expand a shorthand, and check it."""
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
    AdaRound instead learns the rounding from the calibration data, to
    reduce the quantization error.

    Attributes:
        active (bool): Learn the rounding of the weights.
        default_num_iterations (``PositiveInt | None``): How many
            iterations to optimize for. Left out, the AIMET default
            applies.
        default_reg_param (float): The trade-off between the rounding
            loss and the reconstruction loss.
        default_beta_range (tuple[int, int]): The start and the end of
            the beta annealing.
        default_warm_start (float): The share of the iterations during
            which the rounding loss has no effect.

    """

    active: bool = False
    default_num_iterations: PositiveInt | None = None
    default_reg_param: float = 0.01
    default_beta_range: tuple[int, int] = (20, 2)
    default_warm_start: float = 0.2


class AIMETConfig(BaseModelExtraForbid):
    """Quantization with `AIMET
    <https://quic.github.io/aimet-pages/releases/latest/index.html>`_.

    ``luxonis_train quantize`` runs a post-training quantization on the
    validation view, then a quantization-aware training, and exports
    the result.

    Attributes:
        active (bool): Quantize with AIMET. It also adds the AIMET
            callback to the training.
        default_output_bw (``Literal[4, 8, 16]``): The bit width of the
            activations.
        default_param_bw (``Literal[4, 8, 16]``): The bit width of the
            parameters.
        default_data_type (``Literal["int", "float"]``): The data type
            of a quantized value.
        quant_scheme (``Literal["min_max", "tf", "tf_enhanced"]``): How
            AIMET chooses the quantization ranges.
        config (``Params | None``): Extra AIMET settings, inline or as
            the path of a JSON file. See the `AIMET documentation
            <https://quic.github.io/aimet-pages/releases/latest/techniques/runtime_config.html>`_.
        max_calibration_images (``PositiveInt | None``): How many
            validation images calibrate the quantization. It takes the
            first images of the view. Left out, it uses the whole view.
            It is independent of ``trainer.n_validation_batches``.
        fold_batch_norms (bool): Fold the batch normalization layers
            into the preceding layers. This happens before the
            quantization, or after the re-estimation when
            ``batch_norm_reestimation`` is set.
        cross_layer_equalization (bool): Balance the weight ranges
            across consecutive layers before the quantization.
        batch_norm_reestimation (bool): Re-estimate the batch
            normalization statistics after the quantization-aware
            training. Without ``config``, it selects the per-channel
            AIMET config.
        sequential_mse (bool): Optimize the quantization of each layer
            against the output of the float model.
        adaround (AdaroundConfig): Adaptive rounding of the weights.
        epochs (``NonNegativeInt``): How many epochs of
            quantization-aware training to run.
        optimizer (``ConfigItem``): The optimizer of the
            quantization-aware training.
        scheduler (``ConfigItem``): The scheduler of the
            quantization-aware training.

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
        """Load ``config`` from a JSON file when it is given as a path.

        The path may be local or remote; ``LuxonisFileSystem`` of
        ``luxonis_ml`` reads it. A dictionary passes through.

        Args:
            value (``ParamValue``): The raw value of the ``config``
                field.

        Returns:
            ``Any``: The parsed JSON for a path, otherwise ``value``.

        Raises:
            ValueError: When the file cannot be read or parsed.

        """
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
        """Dump an enum member by its name.

        ``default_data_type`` and ``quant_scheme`` are strings in the
        schema, so a string passes through. An ``Enum`` member assigned
        after validation dumps as its ``name``.

        Args:
            value (``Any``): The value of the field.

        Returns:
            str: The ``name`` of an ``Enum`` member, otherwise ``value``
            unchanged.

        """
        if isinstance(value, Enum):
            return value.name
        return value


class ExportConfig(ArchiveConfig):
    """How the trained model is exported and converted.

    `LuxonisModel.export` writes an ONNX file and a ``modelconverter``
    YAML file beside it. `LuxonisModel.convert` then archives the ONNX
    file and runs the conversions of ``blobconverter`` and ``hubai``
    that are active.

    Attributes:
        name (str | None): The name of the exported files, without the
            suffix. Without it, ``model.name`` names the files.
        upload_to_run (bool): Attach the exported files to the tracked
            run.
        upload_url (str | None): A remote location to upload the
            exported files to as well.
        input_shape (list[int] | None): Not read by the export, which
            takes the input shape from the loader.
        quantization_mode (str): The precision the conversion targets.
            One of ``INT8_STANDARD``, ``INT8_ACCURACY_FOCUSED``,
            ``INT8_INT16_MIXED``, ``INT8_INT16_MIXED_ACCURACY_FOCUSED``,
            ``FP16_STANDARD``, or ``FP32_STANDARD``, in any case.
            ``FP16`` and ``FP32`` are short for the standard modes. The
            YAML key ``data_type`` is an alias.
        reverse_input_channels (bool | None): Swap the channel order in
            the ``.blob`` export. Left out, it is ``True`` when
            ``color_space`` is ``RGB``.
        scale_values (list[float] | None): The scale of the input
            normalization, per channel. A single number applies to all
            three channels. Left out, it comes from
            ``trainer.preprocessing.normalize``, scaled by 255, or
            stays ``None`` when ``normalize`` is inactive.
        mean_values (list[float] | None): The mean of the input
            normalization, per channel. A single number applies to all
            three channels. Left out, it comes from
            ``trainer.preprocessing.normalize``, scaled by 255, or
            stays ``None`` when ``normalize`` is inactive.
        onnx (OnnxExportConfig): The options of the ONNX export.
        blobconverter (BlobconverterExportConfig): Conversion to
            ``.blob``, which is deprecated.
        hubai (HubAIExportConfig): Conversion through the HubAI SDK.
        aimet (AIMETConfig): Quantization with AIMET.

    Example:
        >>> ExportConfig(quantization_mode="fp16").quantization_mode
        'FP16_STANDARD'
        >>> ExportConfig(data_type="FP32").quantization_mode
        'FP32_STANDARD'

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
        """Expand a single number to three channel values.

        Args:
            values (``ParamValue``): The raw value of ``scale_values``
                or ``mean_values``.

        Returns:
            ``Any``: A three-element list for a number, otherwise
            ``values`` unchanged.

        Example:
            >>> ExportConfig(scale_values=255).scale_values
            [255.0, 255.0, 255.0]

        """
        if isinstance(values, float | int):
            return [values] * 3
        return values


class StorageConfig(BaseModelExtraForbid):
    """Where Optuna keeps the study.

    Optuna reaches the database through SQLAlchemy, so ``backend`` is
    any driver name `SQLAlchemy
    <https://docs.sqlalchemy.org/en/latest/core/engines.html#database-urls>`_
    accepts. `Config.check_tune_storage` fills in the defaults: with
    ``sqlite``, a missing ``database`` becomes ``study_local.db``; with
    ``postgresql``, a field left out is read from ``POSTGRES_USER``,
    ``POSTGRES_PASSWORD``, ``POSTGRES_HOST``, ``POSTGRES_PORT``, and
    ``POSTGRES_DB``.

    Attributes:
        active (bool): Keep the study in a database, so it survives the
            process. Left false, the study lives in memory.
        backend (str): The SQLAlchemy driver name.
        username (str | None): The user of the database.
        password (``SecretStr | None``): The password of the database.
        host (str | None): The host of the database.
        port (``PositiveInt | None``): The port of the database.
        database (str | None): The name of the database, or the file
            path for ``sqlite``.

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

    Each trial trains a copy of the config with the sampled values.
    The tuner removes the callbacks that would upload, export, archive,
    or test the model of a single trial: ``UploadCheckpoint``,
    ``ExportOnTrainEnd``, ``ArchiveOnTrainEnd``, and
    ``TestOnTrainEnd``. ``ConvertOnTrainEnd`` stays.

    Attributes:
        study_name (str): The name of the study.
        continue_existing_study (bool): Add the trials to a study of
            the same name, instead of starting over.
        use_pruner (bool): Stop a trial early once its result falls
            behind the median of the earlier trials.
        n_trials (``PositiveInt | None``): How many trials this process
            runs. Left out, it runs until ``timeout``.
        timeout (``PositiveInt | None``): How many seconds the study
            runs for. Left out, only ``n_trials`` limits it.
        storage (StorageConfig): Where Optuna keeps the study.
        params (dict[str, list[str | int | float | bool | list]]): The
            parameters to search, and the range of each.

            A key is the path of a config field with a type suffix, as
            in ``trainer.optimizer.params.lr_float``. The suffix
            selects the `Optuna
            <https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html>`_
            method that samples the value:

            - ``categorical``: a list of choices.
            - ``int``: ``[low, high]`` or ``[low, high, step]``, as
              integers.
            - ``float``: ``[low, high]`` or ``[low, high, step]``, as
              floats.
            - ``uniform``: ``[low, high]`` as floats.
            - ``loguniform``: ``[low, high]`` as floats, sampled on a
              logarithmic scale.
            - ``subset``: only for
              ``trainer.preprocessing.augmentations``. Give a list of
              augmentation names and a count. Each trial activates
              that many of them and deactivates the others in the
              list. ``Normalize`` and an unknown name are left out of
              the sample, with a warning.
        monitor (``Literal["metric", "loss"]``): Judge a trial on the
            validation loss and minimize it, or on the main metric and
            maximize it.

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

    Every section has defaults. A training run needs at least the
    ``model`` section, because the default holds no nodes.

    Attributes:
        rich_logging (bool): Render the logs and the progress bar with
            ``rich``.
        model (ModelConfig): The model graph.
        loader (LoaderConfig): Where the data comes from.
        tracker (TrackerConfig): Where the metrics and the artifacts
            go.
        trainer (TrainerConfig): How the model trains.
        exporter (ExportConfig): How the trained model is exported and
            converted.
        archiver (ArchiveConfig): How the NN Archive is named and
            uploaded.
        tuner (TunerConfig): The hyperparameter search that
            ``luxonis_train tune`` runs.
        version (``SemanticVersion``): The schema version the file was
            written for. `get_config` reads it to migrate an older
            file, and ``luxonis_train upgrade config`` rewrites the
            file. `get_config` drops the older key ``config_version``
            with a log line; a direct ``Config(...)`` call accepts it
            as an alias. It dumps as a string.
        ENVIRON (``Environ``): The environment variables, read from
            the process environment and a ``.env`` file.
            ``model_dump`` leaves it out.

            Do not set it in a config file. `check_environment` warns,
            because a secret in a config file is a security risk. Set a
            real environment variable, or use a ``.env`` file.

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
        """Dump the config as a dictionary, without ``ENVIRON``.

        Args:
            exclude (set[str] | None): Extra top-level fields to leave
                out.
            **kwargs (``Any``): Further arguments of
                ``pydantic.BaseModel.model_dump``.

        Returns:
            ``dict[str, Any]``: The config, without ``ENVIRON`` and the
            excluded fields.

        """
        exclude = exclude or set()
        return super().model_dump(exclude=exclude | {"ENVIRON"}, **kwargs)

    @override
    def model_dump_json(
        self, exclude: set[str] | None = None, **kwargs
    ) -> str:
        """Dump the config as a JSON string, without ``ENVIRON``.

        Args:
            exclude (set[str] | None): Extra top-level fields to leave
                out.
            **kwargs (``Any``): Further arguments of
                ``pydantic.BaseModel.model_dump_json``.

        Returns:
            str: The config as JSON, without ``ENVIRON`` and the
            excluded fields.

        """
        exclude = exclude or set()
        return super().model_dump_json(exclude=exclude | {"ENVIRON"}, **kwargs)

    @model_validator(mode="before")
    @classmethod
    def check_environment(cls, data: Params) -> Params:
        """Log a warning when the file holds an ``ENVIRON`` section.

        Args:
            data (``Params``): The raw config dictionary.

        Returns:
            ``Params``: ``data``, unchanged.

        """
        if "ENVIRON" in data:
            logger.warning(
                "Specifying `ENVIRON` section in config file is not "
                "recommended due to security reasons. "
                "Please use environment variables or `.env` file instead."
            )
        return data

    @model_validator(mode="after")
    def check_tune_storage(self) -> Self:
        """Fill in the defaults of the tuner storage.

        With the ``sqlite`` backend, a missing ``database`` becomes
        ``study_local.db``, with a warning. With the ``postgresql``
        backend, each field left out is read from the matching
        ``POSTGRES_*`` variable of ``ENVIRON``. An inactive storage
        stays as it is.

        Returns:
            ``Self``: This instance, with the storage defaults filled
            in.

        """
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
        """Set up the logging before the sections are validated.

        The validators of the sections log warnings, so the logger must
        exist first. The call configures the global ``loguru`` logger
        with or without ``rich``. It skips the setup when the import of
        `luxonis_train.utils` fails.

        Args:
            data (``Params``): The raw config dictionary.

        Returns:
            ``Params``: ``data``, unchanged.

        Raises:
            TypeError: When ``rich_logging`` is not a boolean.

        """
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
        """Load a config from a file, a URL, or a dictionary.

        It downloads a remote path into ``.cache/luxonis_train/``
        first. It accepts any location ``LuxonisFileSystem`` of
        ``luxonis_ml`` reads, such as S3, GCS, or ``mlflow://``. It then
        migrates the data to the current schema with
        `upgrade_config <luxonis_train.upgrade.upgrade_config>`, merges
        the overrides in, and validates the instance. An ``mlflow://``
        source also sets ``tracker.project_id`` and ``tracker.run_id``
        to the run the config came from. When
        ``trainer.smart_cfg_auto_populate`` is set, `smart_auto_populate`
        runs last.

        Args:
            cfg (``PathType | Params | None``): The path or URL of a
                YAML or JSON file, or a dictionary. ``None`` starts from
                the defaults.
            overrides (``Params | list[str] | tuple[str, ...] | None``):
                Values that replace the loaded ones, keyed by the dotted
                path of a field, as in ``trainer.epochs``. A list or a
                tuple alternates keys and values, as the CLI passes
                them. A string value is parsed as a Python literal when
                possible.

        Returns:
            Config: The validated config.

        Raises:
            ValueError: When both ``cfg`` and ``overrides`` are
                ``None``, when a list of overrides has an odd length,
                or when an override names an invalid path.

        """
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
        """Fill in the fields a config leaves out, with a log line each.

        The rules run in this order:

        - ``Mosaic4`` without ``out_width`` or ``out_height`` takes
          both from ``trainer.preprocessing.train_image_size``.
        - When ``train_view``, ``val_view``, and ``test_view`` are the
          same and ``trainer.n_validation_batches`` is unset, it
          becomes ``10``, so validation does not run over the whole
          training set. The rule keeps an explicit value, ``-1``
          included, and warns instead.
        - A predefined model gets ``trainer.accumulate_grad_batches``
          of ``max(1, 64 // batch_size)``, unless it is set. For the
          ``DetectionModel``, ``InstanceSegmentationModel``, and
          ``KeypointDetectionModel`` families, the ``loss_params`` of
          the model get fixed loss weights scaled by that factor. The
          last two families also get the accumulation schedule
          ``{0: 1, 1: (1 + n) // 2, 2: n}``, where ``n`` is the
          factor, when a ``GradientAccumulationScheduler`` callback is
          present.
        - ``UploadCheckpoint``, ``TestOnTrainEnd``, and
          ``ConvertOnTrainEnd`` are added when they are missing.

        `get_config` calls it when ``trainer.smart_cfg_auto_populate``
        is set.

        Returns:
            ``Self``: This instance, with the fields filled in.

        Raises:
            ValueError: When the ``loss_params`` of the predefined
                model is not a dictionary.

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
        """Scale the loss weights of a family into ``loss_params``.

        Return the gradient accumulation schedule of the family, or
        ``None`` when the family has none.

        """
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
