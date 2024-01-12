from abc import ABC, abstractproperty

from luxonis_ml.utils.registry import AutoRegisterMeta

from luxonis_train.utils.config import (
    AttachedModuleConfig,
    LossModuleConfig,
    MetricModuleConfig,
    ModelNodeConfig,
)
from luxonis_train.utils.registry import MODELS


class BasePredefinedModel(
    ABC,
    metaclass=AutoRegisterMeta,
    registry=MODELS,
    register=False,
):
    @abstractproperty
    def nodes(self) -> list[ModelNodeConfig]:
        ...

    @abstractproperty
    def losses(self) -> list[LossModuleConfig]:
        ...

    @abstractproperty
    def metrics(self) -> list[MetricModuleConfig]:
        ...

    @abstractproperty
    def visualizers(self) -> list[AttachedModuleConfig]:
        ...

    def generate_model(
        self,
        include_nodes: bool = True,
        include_losses: bool = True,
        include_metrics: bool = True,
        include_visualizers: bool = True,
    ) -> tuple[
        list[ModelNodeConfig],
        list[LossModuleConfig],
        list[MetricModuleConfig],
        list[AttachedModuleConfig],
    ]:
        nodes = self.nodes if include_nodes else []
        losses = self.losses if include_losses else []
        metrics = self.metrics if include_metrics else []
        visualizers = self.visualizers if include_visualizers else []

        return nodes, losses, metrics, visualizers
