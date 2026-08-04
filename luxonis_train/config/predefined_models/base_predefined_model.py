from abc import abstractmethod
from typing import Literal, cast

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


def model_family_name(
    cls: "type[BasePredefinedModel]", class_name: str | None = None
) -> str:
    """Return the stable registry family for a predefined model class.

    Breaking versions are named C{FamilyV2}, C{FamilyV3}, and so on,
    while users continue to address them as C{Family}.
    """
    name = class_name or cls.__name__
    version_suffix = f"V{cls._VERSION}"
    if name.endswith(version_suffix):
        return name[: -len(version_suffix)]
    return name


class PredefinedModelMeta(VariantMeta):
    """Registers predefined models under versioned C{Family:vN} keys.

    C{AutoRegisterMeta} registers every subclass under its plain class
    name. That entry is replaced here by two: the canonical
    C{Family:vN} key that L{luxonis_train.config.predefined_versions}
    resolves against, and a plain C{Family} alias pointing at the most
    recently registered version, so that looking a model up by its class
    name keeps working.

    Keying happens when the class is created rather than in a one-shot
    sweep after import, so classes registered later - custom models
    loaded through C{--source}, most importantly - are keyed the same
    way. That lets them both add a new version of a shipped family and
    override an existing one, which is what registering under a
    built-in's name did before versioning was introduced.
    """

    def __new__(
        cls,
        name: str,
        bases: tuple[type, ...],
        attrs: dict[str, type],
        register: bool = True,
        register_name: str | None = None,
        registry: Registry | None = None,
    ):
        new_class = super().__new__(
            cls,
            name,
            bases,
            attrs,
            register=register,
            register_name=register_name,
            registry=registry,
        )
        if not register:
            return new_class

        registry = registry if registry is not None else new_class.REGISTRY
        registry._module_dict.pop(register_name or name, None)

        # Abstract intermediates (`SimplePredefinedModel`) cannot be
        # instantiated from a config and must not claim a family name.
        if getattr(new_class, "__abstractmethods__", None):
            return new_class

        model_cls = cast("type[BasePredefinedModel]", new_class)
        family = model_family_name(model_cls, register_name or name)
        registry[f"{family}:v{model_cls._VERSION}"] = model_cls
        registry[family] = model_cls
        return new_class


class BasePredefinedModel(
    VariantBase, metaclass=PredefinedModelMeta, registry=MODELS, register=False
):
    _VERSION: int = 1
    """Version marker for this predefined-model class.

    Subclasses that introduce breaking architecture changes should
    increment this to 2, 3, ... . Registration composes the registry key
    as ``f"{family}:v{cls._VERSION}"`` in
    :mod:`luxonis_train.config.predefined_models` at import time.
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
