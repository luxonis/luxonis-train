from abc import ABC, abstractmethod
from typing import Literal

from loguru import logger
from luxonis_ml.typing import Kwargs, Params, check_type
from luxonis_ml.utils.registry import AutoRegisterMeta
from typeguard import typechecked
from typing_extensions import override

from luxonis_train.config import (
    AttachedModuleConfig,
    LossModuleConfig,
    MetricModuleConfig,
    NodeConfig,
)
from luxonis_train.config.config import FreezingConfig
from luxonis_train.registry import MODELS


class BasePredefinedModel(
    ABC, metaclass=AutoRegisterMeta, registry=MODELS, register=False
):
    @classmethod
    def from_variant(
        cls, variant: str | Literal["default", "none"] | None = None, **kwargs
    ) -> "BasePredefinedModel":
        """Creates a model instance from a predefined variant.

        @type variant: str
        @param variant: The name of the variant to use.
        """
        if variant is None or variant == "none":
            return cls(**kwargs)

        default, variants = cls.get_variants()
        if variant == "default":
            variant = default

        if variant not in variants:
            raise ValueError(
                f"Variant '{variant}' is not available. "
                f"Available variants: {list(variants.keys())}."
            )
        params = variants[variant]
        for key in list(params.keys()):
            if key in kwargs:
                logger.info(
                    f"Overriding variant parameter '{key}' with "
                    f"explicitly provided value `{kwargs[key]}`."
                )
                del params[key]

        return cls(**params, **kwargs)

    @property
    @abstractmethod
    def nodes(self) -> list[NodeConfig]: ...

    @property
    @abstractmethod
    def losses(self) -> list[LossModuleConfig]: ...

    @property
    @abstractmethod
    def metrics(self) -> list[MetricModuleConfig]: ...

    @property
    @abstractmethod
    def visualizers(self) -> list[AttachedModuleConfig]: ...

    @staticmethod
    @abstractmethod
    def get_variants() -> tuple[str, dict[str, Params]]:
        """Returns a name of the default varaint and a dictionary of
        available model variants with their parameters.

        The keys are the variant names, and the values are dictionaries
        of parameters which can be used as C{**kwargs} for the
        predefined model constructor.

        @rtype: tuple[str, dict[str, Params]]
        @return: A tuple containing the default variant name and a
            dictionary of available variants with their parameters.
        """

    def generate_model(
        self,
        include_nodes: bool = True,
        include_losses: bool = True,
        include_metrics: bool = True,
        include_visualizers: bool = True,
    ) -> tuple[
        list[NodeConfig],
        list[LossModuleConfig],
        list[MetricModuleConfig],
        list[AttachedModuleConfig],
    ]:
        nodes = self.nodes if include_nodes else []
        losses = self.losses if include_losses else []
        metrics = self.metrics if include_metrics else []
        visualizers = self.visualizers if include_visualizers else []

        return nodes, losses, metrics, visualizers

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
        return FreezingConfig(**freezing)


class SimplePredefinedModel(BasePredefinedModel):
    @typechecked
    def __init__(
        self,
        *,
        backbone: str,
        head: str,
        neck: str | None = None,
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
        task_name: str = "",
        torchmetrics_task: Literal["binary", "multiclass", "multilabel"]
        | None = None,
        per_class_metrics: bool | None = None,
    ):
        self._backbone = backbone
        self._backbone_params = backbone_params or {}
        self._neck = neck
        self._neck_params = neck_params or {}
        self._head = head
        self._head_params = head_params or {}

        self._task_name = task_name
        self._use_neck = use_neck

        self._loss = loss
        self._loss_params = loss_params or {}
        self._metrics = (
            [metrics] if isinstance(metrics, str) else metrics or []
        )
        if main_metric is None:
            if len(self._metrics) == 1:
                main_metric = self._metrics[0]
            else:
                raise ValueError(
                    "If `main_metric` is not provided, there should be "
                    "exactly one metric defined."
                )
        self._main_metric = main_metric
        self._metrics_params = metrics_params or {}

        if torchmetrics_task is not None:
            self._metrics_params["torchmetrics_task"] = torchmetrics_task

        if per_class_metrics is not None:
            self._metrics_params["per_class_metrics"] = per_class_metrics

        self._visualizer = visualizer
        self._visualizer_params = visualizer_params or {}

        self._enable_confusion_matrix = (
            False
            if not confusion_matrix_available
            else enable_confusion_matrix
        )
        self._confusion_matrix_params = confusion_matrix_params or {}

    @property
    @override
    def nodes(self) -> list[NodeConfig]:
        nodes = [
            NodeConfig(
                name=self._backbone,
                params=self._backbone_params,
                freezing=self._get_freezing(self._backbone_params),
            )
        ]
        if self._neck is not None and self._use_neck:
            nodes.append(
                NodeConfig(
                    name=self._neck,
                    params=self._neck_params,
                    inputs=[self._backbone],
                    freezing=self._get_freezing(self._neck_params),
                )
            )
        nodes.append(
            NodeConfig(
                name=self._head,
                params=self._head_params,
                inputs=[
                    self._neck
                    if self._use_neck and self._neck is not None
                    else self._backbone
                ],
                freezing=self._get_freezing(self._head_params),
                task_name=self._task_name,
            )
        )
        return nodes

    @property
    def losses(self) -> list[LossModuleConfig]:
        return [
            LossModuleConfig(
                name=self._loss,
                attached_to=self._head,
                params=self._loss_params,
                weight=1.0,
            )
        ]

    @property
    def metrics(self) -> list[MetricModuleConfig]:
        metrics = []
        for metric in self._metrics:
            metrics.append(
                MetricModuleConfig(
                    name=metric,
                    attached_to=self._head,
                    params=self._metrics_params,
                    is_main_metric=metric == self._main_metric,
                )
            )
        if self._enable_confusion_matrix:
            metrics.append(
                MetricModuleConfig(
                    name="ConfusionMatrix",
                    attached_to=self._head,
                    params=self._confusion_matrix_params,
                    is_main_metric=False,
                )
            )
        return metrics

    @property
    def visualizers(self) -> list[AttachedModuleConfig]:
        if self._visualizer is None:
            return []
        return [
            AttachedModuleConfig(
                name=self._visualizer,
                attached_to=self._head,
                params=self._visualizer_params,
            )
        ]
