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
    """Version encoded in the module path, e.g.
    C{...detection.v2.model}.
    """
    versions = _NAMESPACE_VERSION.findall(module)
    return int(versions[-1]) if versions else None


class PredefinedModelMeta(VariantMeta):
    """Register versioned predefined models.

    The version comes from the C{v<N>} package the class is defined in
    (e.g. C{predefined_models/detection/v2/model.py} registers
    C{DetectionModel:v2}), so versions of a model share the class name.
    Classes defined outside such a namespace use their C{_VERSION}
    attribute instead. The highest version is additionally registered
    under the bare family name and C{<family>:latest}.
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
    _VERSION: int = 1
    """Registry version for this predefined-model class.

    Inferred from the C{v<N>} package the class is defined in; only
    classes defined outside such a namespace need to set it explicitly.
    """

    @property
    @abstractmethod
    def nodes(self) -> list[NodeConfig]: ...

    @staticmethod
    @abstractmethod
    def get_variants() -> tuple[str, dict[str, Params]]:
        """Get a name of the default varaint and a dictionary of
        available model variants with their parameters.

        The keys are the variant names, and the values are dictionaries
        of parameters which can be used as C{**kwargs} for the
        predefined model constructor.

        @rtype: tuple[str, dict[str, Params]]
        @return: A tuple containing the default variant name and a
            dictionary of available variants with their parameters.
        """

    def generate_nodes(
        self,
        include_losses: bool = True,
        include_metrics: bool = True,
        include_visualizers: bool = True,
    ) -> list[NodeConfig]:
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
        self._metrics = (
            [metrics] if isinstance(metrics, str) else metrics or []
        )
        if main_metric is None and self._metrics:
            if len(self._metrics) == 1:
                main_metric = self._metrics[0]
            else:
                raise ValueError(
                    "If `main_metric` is not provided, there should be "
                    "exactly one metric defined."
                )
        self._main_metric = main_metric
        self._metrics_params = metrics_params or {}
        self._per_class_metrics = per_class_metrics

        if torchmetrics_task is not None:
            self._metrics_params["torchmetrics_task"] = torchmetrics_task

        self._visualizer = visualizer
        self._visualizer_params = visualizer_params or {}

        self._enable_confusion_matrix = (
            False
            if not confusion_matrix_available
            else enable_confusion_matrix
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
