from abc import ABC, abstractmethod

from luxonis_ml.utils.registry import AutoRegisterMeta, Registry

from luxonis_train.config import (
    AttachedModuleConfig,
    LossModuleConfig,
    MetricModuleConfig,
    ModelNodeConfig,
)

MODELS: Registry[type["BasePredefinedModel"]] = Registry(name="models")
"""Registry for all models."""


class BasePredefinedModel(
    ABC,
    metaclass=AutoRegisterMeta,
    registry=MODELS,
    register=False,
):
    @property
    @abstractmethod
    def nodes(self) -> list[ModelNodeConfig]: ...

    @property
    @abstractmethod
    def losses(self) -> list[LossModuleConfig]: ...

    @property
    @abstractmethod
    def metrics(self) -> list[MetricModuleConfig]: ...

    @property
    @abstractmethod
    def visualizers(self) -> list[AttachedModuleConfig]: ...

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
