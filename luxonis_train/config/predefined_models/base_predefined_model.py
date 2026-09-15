"""The base classes of the predefined models.

`BasePredefinedModel` is the base of every predefined model.
`SimplePredefinedModel` builds the usual backbone, neck, and head chain,
so a concrete model only declares its components and its variants.
`PredefinedModelMeta` registers each model under a versioned name.

"""

import re
from abc import abstractmethod
from typing import Any, Literal, cast

from luxonis_ml.typing import Kwargs, Params, check_type
from luxonis_ml.utils.registry import Registry
from typeguard import typechecked
from typing_extensions import override

from luxonis_train.config import (
    AttachedModuleConfig,
    LossModuleConfig,
    MetricModuleConfig,
    NodeConfig,
)
from luxonis_train.config.config import FinetuningConfig, FreezingConfig
from luxonis_train.registry import MODELS
from luxonis_train.variants import VariantBase, VariantMeta

_NAMESPACE_VERSION = re.compile(r"\.v(\d+)(?=\.|$)", re.ASCII)


def _namespace_version(module: str) -> int | None:
    """Return the version encoded in the module path.

    The version is the number of the last ``v<N>`` package in the path.

    Args:
        module (str): The dotted path of a module.

    Returns:
        int | None: The number ``N``, or ``None`` when the path has no
        ``v<N>`` package.

    Example:
        >>> _namespace_version("predefined_models.detection.v2.model")
        2
        >>> print(_namespace_version("predefined_models.detection.model"))
        None

    """
    versions = _NAMESPACE_VERSION.findall(module)
    return int(versions[-1]) if versions else None


class PredefinedModelMeta(VariantMeta):
    """Metaclass that registers the predefined models under versioned
    names.

    The version comes from the ``v<N>`` package that defines the class.
    A ``DetectionModel`` class in ``predefined_models/detection/v2/``
    registers as ``DetectionModel:v2``, so every version of a model
    keeps the class name. A class outside such a package uses its
    ``_VERSION`` attribute. The metaclass also registers the highest
    version under the bare family name and under ``<family>:latest``.

    """

    def __new__(
        cls,
        name: str,
        bases: tuple[type, ...],
        attrs: dict[str, Any],
        register: bool = True,
        register_name: str | None = None,
        registry: Registry | None = None,
    ):
        """Create the class and register it under its versioned name.

        When the module path holds a ``v<N>`` package, the metaclass
        stores ``N`` in ``_VERSION``. It skips the registration when
        ``register`` is ``False`` or when the class is still abstract.
        Otherwise it registers the class as ``<family>:v<N>``. The
        family is ``register_name`` or the class name, and ``N`` is
        ``_VERSION``. It also registers the class as ``<family>`` and as
        ``<family>:latest`` in one of these cases:

        - the registry has no ``<family>`` entry;
        - the ``<family>`` entry has no integer ``_VERSION``;
        - the ``<family>`` entry has a lower or equal version.

        Args:
            name (str): The name of the new class.
            bases (``tuple[type, ...]``): The base classes.
            attrs (``dict[str, Any]``): The namespace of the class body.
            register (bool): Register the class. Set it to ``False``
                for a base class.
            register_name (str | None): The family name to register
                under. ``None`` takes the class name.
            registry (``Registry | None``): The registry to use. ``None``
                takes the ``REGISTRY`` attribute of the class.

        Returns:
            type: The new class.

        Raises:
            ValueError: When the class sets ``_VERSION`` to a value that
                differs from the version of its ``v<N>`` package.

        """
        version = _namespace_version(attrs.get("__module__", ""))
        if version is not None:
            explicit = attrs.get("_VERSION")
            if explicit is not None and explicit != version:
                raise ValueError(
                    f"'{name}' sets `_VERSION = {explicit}` but is defined "
                    f"in a 'v{version}' package. Drop the attribute; the "
                    "version is inferred from the package name."
                )
            attrs["_VERSION"] = version
        new_class = super().__new__(
            cls, name, bases, attrs, register=False, registry=registry
        )
        # Abstract intermediates must not claim a family name.
        if not register or getattr(new_class, "__abstractmethods__", None):
            return new_class

        registry = registry if registry is not None else new_class.REGISTRY
        model_cls = cast("type[BasePredefinedModel]", new_class)
        family = register_name or name
        registry[f"{family}:v{model_cls._VERSION}"] = model_cls
        aliased = registry._module_dict.get(family)
        aliased_version = getattr(aliased, "_VERSION", None)
        if (
            not isinstance(aliased_version, int)
            or aliased_version <= model_cls._VERSION
        ):
            registry[family] = model_cls
            registry[f"{family}:latest"] = model_cls
        return new_class


class BasePredefinedModel(
    VariantBase, metaclass=PredefinedModelMeta, registry=MODELS, register=False
):
    """The base class of a predefined model.

    A subclass returns the node graph from `nodes` and declares its
    variants in `get_variants`.
    `luxonis_train.config.config.ModelConfig.validate_predefined_model`
    builds the model from the ``model.predefined_model`` section of a
    config and appends the result of `generate_nodes` to
    ``model.nodes``. Subclass this class directly when the graph is not
    a plain backbone, neck, and head chain. Otherwise, subclass
    `SimplePredefinedModel`.

    `PredefinedModelMeta` registers every concrete subclass in the
    ``MODELS`` registry as ``<name>:v<N>``. It also registers the
    highest version of a family as ``<name>`` and as ``<name>:latest``.

    """

    _VERSION: int = 1
    """The registry version of this class.

    `PredefinedModelMeta` sets it from the ``v<N>`` package that defines
    the class. A class outside such a package inherits the value of its
    base class, unless it sets its own value. On this class, the value
    is ``1``.

    """

    @property
    @abstractmethod
    def nodes(self) -> list[NodeConfig]:
        """The node configs of the model graph.

        An implementation must build new configs on each access, because
        `generate_nodes` edits the configs it receives. Each config
        holds the losses, the metrics, and the visualizers of its node.

        """
        ...

    @staticmethod
    @abstractmethod
    def get_variants() -> tuple[str, dict[str, Params]]:
        """Get the default variant name and the available variants.

        The keys of the dictionary are the variant names. Each value
        holds keyword arguments for the constructor of the model.
        `VariantMeta` passes the arguments of the selected variant to
        ``__init__``, and ``variant="default"`` selects the default
        variant. An argument that the caller also gives replaces the
        variant value as a whole, so `VariantMeta` does not merge a
        dictionary value.

        Returns:
            ``tuple[str, dict[str, Params]]``: The default variant name,
            and the variants with their constructor arguments.

        """

    def generate_nodes(
        self,
        include_losses: bool = True,
        include_metrics: bool = True,
        include_visualizers: bool = True,
    ) -> list[NodeConfig]:
        """Return the node graph with the attached modules filtered.

        `luxonis_train.config.config.ModelConfig.validate_predefined_model`
        calls it with the ``include_*`` flags of the ``predefined_model``
        section. A flag set to ``False`` empties the matching list on
        every node. The method edits the configs from `nodes` in place.

        Args:
            include_losses (bool): Keep the losses of the nodes.
            include_metrics (bool): Keep the metrics of the nodes.
            include_visualizers (bool): Keep the visualizers of the
                nodes.

        Returns:
            list[NodeConfig]: The configs from `nodes`, filtered.

        Example:
            >>> from luxonis_train.config.predefined_models import (
            ...     ClassificationModel,
            ... )
            >>> model = ClassificationModel(variant="light")
            >>> head = model.generate_nodes(include_metrics=False)[-1]
            >>> [loss.name for loss in head.losses]
            ['CrossEntropyLoss']
            >>> head.metrics
            []

        """
        nodes = self.nodes
        for node in nodes:
            if not include_losses:
                node.losses = []
            if not include_metrics:
                node.metrics = []
            if not include_visualizers:
                node.visualizers = []
        return nodes

    @staticmethod
    def _get_freezing(params: Params) -> FreezingConfig:
        """Pop ``freezing`` out of ``params`` and build its config.

        A dictionary becomes an active `FreezingConfig`, unless it sets
        ``active`` itself. A `FreezingConfig` passes through. Without
        the key, the method returns an inactive `FreezingConfig`.

        Args:
            params (``Params``): The constructor parameters of a node.
                The method removes the ``freezing`` key from them.

        Returns:
            FreezingConfig: The freezing config of the node.

        Raises:
            ValueError: When ``freezing`` is neither a dictionary nor a
                `FreezingConfig`.

        """
        if "freezing" not in params:
            return FreezingConfig()
        freezing = params.pop("freezing")
        if isinstance(freezing, FreezingConfig):
            return freezing
        if not check_type(freezing, Kwargs):
            raise ValueError(
                f"`backbone_params.freezing` should be a dictionary, "
                f"got '{freezing}' instead."
            )
        return FreezingConfig(**{"active": True, **freezing})


class SimplePredefinedModel(BasePredefinedModel):
    """A predefined model with a backbone, an optional neck, and a head.

    A subclass names its components and its variants. This class wires
    them into a chain and attaches the loss, the metrics, and the
    visualizer to the head. It also applies the freezing and the
    finetuning that a config asks for. The keys of
    ``model.predefined_model.params`` in a config are the keyword
    arguments of ``__init__``. A ``variant`` key does not reach
    ``__init__``. It selects the variant.

    """

    @typechecked
    def __init__(
        self,
        *,
        backbone: str,
        backbone_variant: str | None = None,
        head: str,
        head_variant: str | None = None,
        neck: str | None = None,
        neck_variant: str | None = None,
        loss: str,
        metrics: str | list[str] | None,
        main_metric: str | None = None,
        visualizer: str | None = None,
        confusion_matrix_available: bool = False,
        backbone_params: Params | None = None,
        neck_params: Params | None = None,
        use_neck: bool = True,
        head_params: Params | None = None,
        loss_params: Params | None = None,
        metrics_params: Params | None = None,
        visualizer_params: Params | None = None,
        enable_confusion_matrix: bool = True,
        confusion_matrix_params: Params | None = None,
        task_name: str | None = None,
        torchmetrics_task: Literal["binary", "multiclass", "multilabel"]
        | None = None,
        per_class_metrics: bool | None = None,
        finetuning: dict[Literal["backbone", "neck", "head"], list[Params]]
        | None = None,
    ):
        """Initialize the model from the names of its components.

        All arguments are keyword-only. The ``typechecked`` decorator
        checks their types at run time.

        Args:
            backbone (str): The class name of the registered backbone
                node.
            backbone_variant (str | None): The variant of the backbone.
                ``None`` builds the backbone without variant
                parameters.
            head (str): The class name of the registered head node.
            head_variant (str | None): The variant of the head.
                ``None`` builds the head without variant parameters.
            neck (str | None): The class name of the registered neck
                node. ``None`` connects the head to the backbone.
            neck_variant (str | None): The variant of the neck.
                ``None`` builds the neck without variant parameters.
            loss (str): The class name of the registered loss. The
                model attaches it to the head with weight ``1.0``.
            metrics (str | list[str] | None): The class names of the
                registered metrics attached to the head. A string
                names one metric. ``None`` attaches no metric.
            main_metric (str | None): The metric to mark as the main
                metric. The trainer keeps the checkpoints with the
                highest values of this metric in ``best_val_metric``.
                ``None`` takes the only name in ``metrics``, or no
                metric when ``metrics`` is empty. A name that is not in
                ``metrics`` marks no metric. When no metric of the
                config is marked, `ModelConfig.check_main_metric` marks
                the first one.
            visualizer (str | None): The class name of the registered
                visualizer attached to the head. ``None`` attaches no
                visualizer.
            confusion_matrix_available (bool): Whether the head
                supports the `ConfusionMatrix` metric. A subclass sets
                it for its head.
            backbone_params (``Params | None``): The constructor
                parameters of the backbone. A ``freezing`` key does not
                reach the constructor. See the notes.
            neck_params (``Params | None``): The constructor parameters
                of the neck, with the same ``freezing`` key.
            use_neck (bool): Build the neck. ``False`` leaves the neck
                out even when ``neck`` is set, and the head reads from
                the backbone.
            head_params (``Params | None``): The constructor parameters
                of the head, with the same ``freezing`` key.
            loss_params (``Params | None``): The constructor parameters
                of the loss.
            metrics_params (``Params | None``): The constructor
                parameters that every metric in ``metrics`` receives.
                The ``ConfusionMatrix`` metric that
                ``enable_confusion_matrix`` adds does not receive them.
            visualizer_params (``Params | None``): The constructor
                parameters of the visualizer.
            enable_confusion_matrix (bool): Attach the
                ``ConfusionMatrix`` metric to the head, without the
                main metric flag. It has no effect when
                ``confusion_matrix_available`` is ``False``.
            confusion_matrix_params (``Params | None``): The constructor
                parameters of the ``ConfusionMatrix`` metric.
            task_name (str | None): The dataset task the head reads. It
                becomes the ``task_name`` of the head node.
            torchmetrics_task (``Literal["binary", "multiclass", "multilabel"] | None``):
                A value for the ``torchmetrics_task`` key that every
                metric in ``metrics`` receives. ``None`` adds no key.
                The key goes into ``metrics_params``, so a non-empty
                ``metrics_params`` dictionary changes in place. No
                metric of this package reads the key. The
                `TorchMetricWrapper` metrics read ``task`` and pass
                ``torchmetrics_task`` on to ``torchmetrics``, which
                raises ``ValueError`` for it. Set ``task`` in
                ``metrics_params`` for them instead.
            per_class_metrics (bool | None): A value for the
                ``per_class_metrics`` key that every metric in
                ``metrics`` receives. ``None`` adds no key. When
                `LuxonisLightningModule` builds a metric, the key
                becomes the per-class parameter that the metric class
                declares. When the class declares none, the module
                drops the key and logs a warning.
            finetuning (``dict[Literal["backbone", "neck", "head"], list[Params]] | None``):
                The finetuning entries of each component. Each
                dictionary becomes a `FinetuningConfig` of that node.

        Raises:
            ValueError: When ``main_metric`` is ``None`` and ``metrics``
                names more than one metric.

        Notes:
            A ``freezing`` key in ``backbone_params``, ``neck_params``,
            or ``head_params`` holds the `FreezingConfig` of that node,
            either as an instance or as a dictionary of its fields. A
            dictionary without ``active`` freezes the node. `nodes`
            removes the key when it builds the configs.

        """
        self._backbone = backbone
        self._backbone_params = backbone_params or {}
        self._backbone_variant = backbone_variant
        self._neck = neck
        self._neck_params = neck_params or {}
        self._neck_variant = neck_variant
        self._head = head
        self._head_params = head_params or {}
        self._head_variant = head_variant
        self._finetuning = finetuning or {}

        self._task_name = task_name
        self._use_neck = use_neck

        self._loss = loss
        self._loss_params = loss_params or {}
        self._set_metrics(
            metrics,
            main_metric,
            metrics_params,
            per_class_metrics,
            torchmetrics_task,
        )
        self._visualizer = visualizer
        self._visualizer_params = visualizer_params or {}

        self._enable_confusion_matrix = (
            confusion_matrix_available and enable_confusion_matrix
        )
        self._confusion_matrix_params = confusion_matrix_params or {}

    def _get_finetuning(
        self, module: Literal["backbone", "neck", "head"]
    ) -> list[FinetuningConfig]:
        return [
            FinetuningConfig(**params)  # type: ignore
            for params in self._finetuning.get(module, [])
        ]

    @property
    @override
    def nodes(self) -> list[NodeConfig]:
        """The backbone, the neck when used, and the head, as configs.

        The backbone has no inputs, so it reads from the loader. The
        neck reads from the backbone. The head reads from the neck, or
        from the backbone when ``use_neck`` is ``False`` or ``neck`` is
        ``None``. Each config carries the ``params``, the ``variant``,
        the ``freezing``, and the ``finetuning`` of its node. The head
        also carries ``task_name``, the loss with weight ``1.0``, the
        metrics, the ``ConfusionMatrix`` metric when it is enabled, and
        the visualizer.

        The property pops the ``freezing`` key from the stored
        parameter dictionaries of the nodes. A second access therefore
        builds nodes that are not frozen.

        Example:
            >>> from luxonis_train.config.predefined_models import (
            ...     DetectionModel,
            ... )
            >>> model = DetectionModel(variant="light")
            >>> [(node.name, node.inputs) for node in model.nodes]
            [('EfficientRep', []),
             ('RepPANNeck', ['EfficientRep']),
             ('EfficientBBoxHead', ['RepPANNeck'])]
            >>> [metric.name for metric in model.nodes[-1].metrics]
            ['MeanAveragePrecision', 'ConfusionMatrix']
            >>> model = DetectionModel(variant="light", use_neck=False)
            >>> [(node.name, node.inputs) for node in model.nodes]
            [('EfficientRep', []), ('EfficientBBoxHead', ['EfficientRep'])]

        """
        metrics = self._generate_metrics()

        nodes = [
            NodeConfig(
                name=self._backbone,
                params=self._backbone_params,
                variant=self._backbone_variant,
                freezing=self._get_freezing(self._backbone_params),
                finetuning=self._get_finetuning("backbone"),
            )
        ]
        if self._neck is not None and self._use_neck:
            nodes.append(
                NodeConfig(
                    name=self._neck,
                    params=self._neck_params,
                    variant=self._neck_variant,
                    inputs=[self._backbone],
                    freezing=self._get_freezing(self._neck_params),
                    finetuning=self._get_finetuning("neck"),
                )
            )
        nodes.append(
            NodeConfig(
                name=self._head,
                params=self._head_params,
                variant=self._head_variant,
                inputs=[
                    self._neck
                    if self._use_neck and self._neck is not None
                    else self._backbone
                ],
                freezing=self._get_freezing(self._head_params),
                task_name=self._task_name,
                losses=[
                    LossModuleConfig(
                        name=self._loss,
                        params=self._loss_params,
                        weight=1.0,
                    )
                ],
                metrics=metrics
                + (
                    [
                        MetricModuleConfig(
                            name="ConfusionMatrix",
                            params=self._confusion_matrix_params,
                            is_main_metric=False,
                        )
                    ]
                    if self._enable_confusion_matrix
                    else []
                ),
                visualizers=[
                    AttachedModuleConfig(
                        name=self._visualizer,
                        params=self._visualizer_params,
                    )
                ]
                if self._visualizer is not None
                else [],
                finetuning=self._get_finetuning("head"),
            )
        )
        return nodes

    def _generate_metrics(self) -> list[MetricModuleConfig]:
        metrics = []
        for metric in self._metrics:
            metric_params = dict(self._metrics_params)
            if self._per_class_metrics is not None:
                metric_params["per_class_metrics"] = self._per_class_metrics

            metrics.append(
                MetricModuleConfig(
                    name=metric,
                    params=metric_params,
                    is_main_metric=metric == self._main_metric,
                )
            )

        return metrics

    def _set_metrics(
        self,
        metrics: str | list[str] | None,
        main_metric: str | None,
        metrics_params: Params | None,
        per_class_metrics: bool | None,
        torchmetrics_task: Literal["binary", "multiclass", "multilabel"]
        | None,
    ) -> None:
        """Store the metric names and the parameters they share.

        Args:
            metrics (str | list[str] | None): The metric names. A string
                becomes a list of one name, and ``None`` becomes an
                empty list.
            main_metric (str | None): The main metric. ``None`` takes the
                only name in ``metrics``, or stays ``None`` when
                ``metrics`` is empty.
            metrics_params (``Params | None``): The parameters every
                metric receives. When ``torchmetrics_task`` is set, a
                non-empty dictionary gains that key in place.
            per_class_metrics (bool | None): The ``per_class_metrics``
                value for every metric. ``None`` adds no key.
            torchmetrics_task (``Literal["binary", "multiclass", "multilabel"] | None``):
                The ``torchmetrics_task`` value for every metric.
                ``None`` adds no key.

        Raises:
            ValueError: When ``main_metric`` is ``None`` and ``metrics``
                names more than one metric.

        """
        self._metrics = (
            [metrics] if isinstance(metrics, str) else metrics or []
        )
        if main_metric is None and self._metrics:
            if len(self._metrics) != 1:
                raise ValueError(
                    "If `main_metric` is not provided, there should be "
                    "exactly one metric defined."
                )
            main_metric = self._metrics[0]
        self._main_metric = main_metric
        self._metrics_params = metrics_params or {}
        self._per_class_metrics = per_class_metrics

        if torchmetrics_task is not None:
            self._metrics_params["torchmetrics_task"] = torchmetrics_task
