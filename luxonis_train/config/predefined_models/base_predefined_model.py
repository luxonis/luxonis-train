from abc import ABC, abstractmethod

from luxonis_ml.typing import Kwargs, Params, check_type
from luxonis_ml.utils.registry import AutoRegisterMeta

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
