from dataclasses import dataclass, field
from typing import Literal

from luxonis_train.config import (
    AttachedModuleConfig,
    LossModuleConfig,
    MetricModuleConfig,
    ModelNodeConfig,
)
from luxonis_train.utils import Kwargs

from .base_predefined_model import BasePredefinedModel


@dataclass
class KeypointDetectionModel(BasePredefinedModel):
    use_neck: bool = True
    backbone_params: Kwargs = field(default_factory=dict)
    neck_params: Kwargs = field(default_factory=dict)
    head_params: Kwargs = field(default_factory=dict)
    loss_params: Kwargs = field(default_factory=dict)
    head_type: Literal[
        "ImplicitKeypointBBoxHead", "EfficientKeypointBBoxHead"
    ] = "EfficientKeypointBBoxHead"
    kpt_visualizer_params: Kwargs = field(default_factory=dict)
    bbox_visualizer_params: Kwargs = field(default_factory=dict)
    bbox_task_name: str | None = None
    kpt_task_name: str | None = None

    @property
    def nodes(self) -> list[ModelNodeConfig]:
        nodes = [
            ModelNodeConfig(
                name="EfficientRep",
                alias="kpt_detection_backbone",
                freezing=self.backbone_params.pop("freezing", {}),
                params=self.backbone_params,
            ),
        ]
        if self.use_neck:
            nodes.append(
                ModelNodeConfig(
                    name="RepPANNeck",
                    alias="kpt_detection_neck",
                    inputs=["kpt_detection_backbone"],
                    freezing=self.neck_params.pop("freezing", {}),
                    params=self.neck_params,
                )
            )

        task = {}
        if self.bbox_task_name is not None:
            task["boundingbox"] = self.bbox_task_name
        if self.kpt_task_name is not None:
            task["keypoints"] = self.kpt_task_name

        nodes.append(
            ModelNodeConfig(
                name=self.head_type,
                alias="kpt_detection_head",
                inputs=["kpt_detection_neck"]
                if self.use_neck
                else ["kpt_detection_backbone"],
                freezing=self.head_params.pop("freezing", {}),
                params=self.head_params,
                task=task,
            )
        )
        return nodes

    @property
    def losses(self) -> list[LossModuleConfig]:
        return [
            LossModuleConfig(
                name=self.head_type.replace("Head", "Loss"),
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
                alias="kpt_detection_oks",
                attached_to="kpt_detection_head",
                is_main_metric=True,
            ),
            MetricModuleConfig(
                name="MeanAveragePrecisionKeypoints",
                alias="kpt_detection_map",
                attached_to="kpt_detection_head",
            ),
        ]

    @property
    def visualizers(self) -> list[AttachedModuleConfig]:
        return [
            AttachedModuleConfig(
                name="MultiVisualizer",
                alias="kpt_detection_visualizer",
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
