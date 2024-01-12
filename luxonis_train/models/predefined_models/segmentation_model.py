from dataclasses import dataclass, field
from typing import Literal

from luxonis_train.utils.config import (
    AttachedModuleConfig,
    LossModuleConfig,
    MetricModuleConfig,
    ModelNodeConfig,
)
from luxonis_train.utils.types import Kwargs

from .base_predefined_model import BasePredefinedModel


@dataclass
class SegmentationModel(BasePredefinedModel):
    backbone: str = "MicroNet"
    task: Literal["binary", "multiclass"] = "binary"
    backbone_params: Kwargs = field(default_factory=dict)
    head_params: Kwargs = field(default_factory=dict)
    loss_params: Kwargs = field(default_factory=dict)
    visualizer_params: Kwargs = field(default_factory=dict)

    @property
    def nodes(self) -> list[ModelNodeConfig]:
        return [
            ModelNodeConfig(
                name=self.backbone,
                override_name="segmentation_backbone",
                frozen=self.backbone_params.pop("frozen", False),
                params=self.backbone_params,
            ),
            ModelNodeConfig(
                name="SegmentationHead",
                override_name="segmentation_head",
                inputs=["segmentation_backbone"],
                frozen=self.head_params.pop("frozen", False),
                params=self.head_params,
            ),
        ]

    @property
    def losses(self) -> list[LossModuleConfig]:
        return [
            LossModuleConfig(
                name="BCEWithLogitsLoss"
                if self.task == "binary"
                else "CrossEntropyLoss",
                override_name="segmentation_loss",
                attached_to="segmentation_head",
                params=self.loss_params,
                weight=1.0,
            )
        ]

    @property
    def metrics(self) -> list[MetricModuleConfig]:
        return [
            MetricModuleConfig(
                name="JaccardIndex",
                override_name="segmentation_jaccard_index",
                attached_to="segmentation_head",
                is_main_metric=True,
                params={"task": self.task},
            ),
            MetricModuleConfig(
                name="F1Score",
                override_name="segmentation_f1_score",
                attached_to="segmentation_head",
                params={"task": self.task},
            ),
        ]

    @property
    def visualizers(self) -> list[AttachedModuleConfig]:
        return [
            AttachedModuleConfig(
                name="SegmentationVisualizer",
                override_name="segmentation_visualizer",
                attached_to="segmentation_head",
                params=self.visualizer_params,
            )
        ]
