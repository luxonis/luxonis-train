from dataclasses import dataclass, field

from luxonis_train.utils.config import (
    AttachedModuleConfig,
    LossModuleConfig,
    MetricModuleConfig,
    ModelNodeConfig,
)
from luxonis_train.utils.types import Kwargs

from .base_predefined_model import BasePredefinedModel


@dataclass
class DetectionModel(BasePredefinedModel):
    use_neck: bool = True
    backbone_params: Kwargs = field(default_factory=dict)
    neck_params: Kwargs = field(default_factory=dict)
    head_params: Kwargs = field(default_factory=dict)
    loss_params: Kwargs = field(default_factory=dict)
    visualizer_params: Kwargs = field(default_factory=dict)

    @property
    def nodes(self) -> list[ModelNodeConfig]:
        nodes = [
            ModelNodeConfig(
                name="EfficientRep",
                override_name="detection_backbone",
                frozen=self.backbone_params.pop("frozen", False),
                params=self.backbone_params,
            ),
        ]
        if self.use_neck:
            nodes.append(
                ModelNodeConfig(
                    name="RepPANNeck",
                    override_name="detection_neck",
                    inputs=["detection_backbone"],
                    frozen=self.neck_params.pop("frozen", False),
                    params=self.neck_params,
                )
            )

        nodes.append(
            ModelNodeConfig(
                name="EfficientBBoxHead",
                override_name="detection_head",
                frozen=self.head_params.pop("frozen", False),
                inputs=["detection_neck"] if self.use_neck else ["detection_backbone"],
                params=self.head_params,
            )
        )
        return nodes

    @property
    def losses(self) -> list[LossModuleConfig]:
        return [
            LossModuleConfig(
                name="AdaptiveDetectionLoss",
                override_name="detection_loss",
                attached_to="detection_head",
                params=self.loss_params,
                weight=1.0,
            )
        ]

    @property
    def metrics(self) -> list[MetricModuleConfig]:
        return [
            MetricModuleConfig(
                name="MeanAveragePrecision",
                override_name="detection_map",
                attached_to="detection_head",
                is_main_metric=True,
            ),
        ]

    @property
    def visualizers(self) -> list[AttachedModuleConfig]:
        return [
            AttachedModuleConfig(
                name="BBoxVisualizer",
                override_name="detection_visualizer",
                attached_to="detection_head",
                params=self.visualizer_params,
            )
        ]
