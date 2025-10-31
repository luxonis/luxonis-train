from abc import abstractmethod
from typing import Literal

from luxonis_ml.typing import Kwargs, Params, check_type
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
from luxonis_train.variants import VariantBase


class BasePredefinedModel(VariantBase, registry=MODELS, register=False):
    @property
    @abstractmethod
    def nodes(self) -> list[NodeConfig]: ...

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
                variant=self._backbone_variant,
                freezing=self._get_freezing(self._backbone_params),
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
                metrics=[
                    MetricModuleConfig(
                        name=metric,
                        params=self._metrics_params,
                        is_main_metric=metric == self._main_metric,
                    )
                    for metric in self._metrics
                ]
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
            )
        )
        return nodes
