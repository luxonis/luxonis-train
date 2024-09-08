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
class OBBDetectionModel(BasePredefinedModel):
    use_neck: bool = True
    backbone_params: Kwargs = field(default_factory=dict)
    neck_params: Kwargs = field(default_factory=dict)
    head_params: Kwargs = field(default_factory=dict)
    loss_params: Kwargs = field(default_factory=dict)
    visualizer_params: Kwargs = field(default_factory=dict)
    task_name: str | None = None

    @property
    def nodes(self) -> list[ModelNodeConfig]:
        nodes = [
            ModelNodeConfig(
                name="EfficientRep",
                alias="detection_backbone",
                freezing=self.backbone_params.pop("freezing", {}),
                params=self.backbone_params,
            ),
        ]
        if self.use_neck:
            nodes.append(
                ModelNodeConfig(
                    name="RepPANNeck",
                    alias="detection_neck",
                    inputs=["detection_backbone"],
                    freezing=self.neck_params.pop("freezing", {}),
                    params=self.neck_params,
                )
            )

        nodes.append(
            ModelNodeConfig(
                name="EfficientOBBoxHead",
                alias="detection_obb_head",
                freezing=self.head_params.pop("freezing", {}),
                inputs=["detection_neck"] if self.use_neck else ["detection_backbone"],
                params=self.head_params,
                task=self.task_name,
            )
        )
        return nodes

    @property
    def losses(self) -> list[LossModuleConfig]:
        return [
            LossModuleConfig(
                name="OBBDetectionLoss",
                alias="detection_obb_loss",
                attached_to="detection_obb_head",
                params=self.loss_params,
                weight=1.0,
            )
        ]

    @property
    def metrics(self) -> list[MetricModuleConfig]:
        return [
            MetricModuleConfig(
                name="MeanAveragePrecisionOBB",
                alias="detection_map_obb",
                attached_to="detection_obb_head",
                is_main_metric=True,
            ),
        ]

    @property
    def visualizers(self) -> list[AttachedModuleConfig]:
        return [
            AttachedModuleConfig(
                name="OBBoxVisualizer",
                alias="detection_visualizer_obb",
                attached_to="detection_obb_head",
                params=self.visualizer_params,
            )
        ]
