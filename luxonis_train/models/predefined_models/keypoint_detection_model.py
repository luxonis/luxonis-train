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
class KeypointDetectionModel(BasePredefinedModel):
    use_neck: bool = True
    backbone_params: Kwargs = field(default_factory=dict)
    neck_params: Kwargs = field(default_factory=dict)
    head_params: Kwargs = field(default_factory=dict)
    loss_params: Kwargs = field(default_factory=dict)
    kpt_visualizer_params: Kwargs = field(default_factory=dict)
    bbox_visualizer_params: Kwargs = field(default_factory=dict)

    @property
    def nodes(self) -> list[ModelNodeConfig]:
        nodes = [
            ModelNodeConfig(
                name="EfficientRep",
                override_name="kpt_detection_backbone",
                frozen=self.backbone_params.pop("frozen", False),
                params=self.backbone_params,
            ),
        ]
        if self.use_neck:
            nodes.append(
                ModelNodeConfig(
                    name="RepPANNeck",
                    override_name="kpt_detection_neck",
                    inputs=["kpt_detection_backbone"],
                    frozen=self.neck_params.pop("frozen", False),
                    params=self.neck_params,
                )
            )

        nodes.append(
            ModelNodeConfig(
                name="ImplicitKeypointBBoxHead",
                override_name="kpt_detection_head",
                inputs=["kpt_detection_neck"]
                if self.use_neck
                else ["kpt_detection_backbone"],
                frozen=self.head_params.pop("frozen", False),
                params=self.head_params,
            )
        )
        return nodes

    @property
    def losses(self) -> list[LossModuleConfig]:
        return [
            LossModuleConfig(
                name="ImplicitKeypointBBoxLoss",
                attached_to="kpt_detection_head",
                params=self.loss_params,
                weight=1.0,
            )
        ]

    @property
    def metrics(self) -> list[MetricModuleConfig]:
        return [
            MetricModuleConfig(
                name="ObjectKeypointSimilarity",
                override_name="kpt_detection_oks",
                attached_to="kpt_detection_head",
                is_main_metric=True,
            ),
            MetricModuleConfig(
                name="MeanAveragePrecisionKeypoints",
                override_name="kpt_detection_map",
                attached_to="kpt_detection_head",
            ),
        ]

    @property
    def visualizers(self) -> list[AttachedModuleConfig]:
        return [
            AttachedModuleConfig(
                name="MultiVisualizer",
                override_name="kpt_detection_visualizer",
                attached_to="kpt_detection_head",
                params={
                    "visualizers": [
                        {
                            "name": "KeypointVisualizer",
                            "params": self.kpt_visualizer_params,
                        },
                        {
                            "name": "BBoxVisualizer",
                            "params": self.bbox_visualizer_params,
                        },
                    ]
                },
            )
        ]
