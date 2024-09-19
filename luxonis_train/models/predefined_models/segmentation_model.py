from dataclasses import dataclass, field
from typing import Literal

from luxonis_train.utils import Kwargs
from luxonis_train.utils.config import (
    AttachedModuleConfig,
    LossModuleConfig,
    MetricModuleConfig,
    ModelNodeConfig,
)

from .base_predefined_model import BasePredefinedModel


@dataclass
class SegmentationModel(BasePredefinedModel):
    backbone: str = "MicroNet"
    task: Literal["binary", "multiclass"] = "binary"
    backbone_params: Kwargs = field(default_factory=dict)
    head_params: Kwargs = field(default_factory=dict)
    loss_params: Kwargs = field(default_factory=dict)
    visualizer_params: Kwargs = field(default_factory=dict)
    task_name: str | None = None

    @property
    def nodes(self) -> list[ModelNodeConfig]:
        return [
            ModelNodeConfig(
                name=self.backbone,
                alias="segmentation_backbone",
                freezing=self.backbone_params.pop("freezing", {}),
                params=self.backbone_params,
            ),
            ModelNodeConfig(
                name="SegmentationHead",
                alias="segmentation_head",
                inputs=["segmentation_backbone"],
                freezing=self.head_params.pop("freezing", {}),
                params=self.head_params,
                task=self.task_name,
            ),
        ]

    @property
    def losses(self) -> list[LossModuleConfig]:
        return [
            LossModuleConfig(
                name="BCEWithLogitsLoss"
                if self.task == "binary"
                else "CrossEntropyLoss",
                alias="segmentation_loss",
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
                alias="segmentation_jaccard_index",
                attached_to="segmentation_head",
                is_main_metric=True,
                params={"task": self.task},
            ),
            MetricModuleConfig(
                name="F1Score",
                alias="segmentation_f1_score",
                attached_to="segmentation_head",
                params={"task": self.task},
            ),
        ]

    @property
    def visualizers(self) -> list[AttachedModuleConfig]:
        return [
            AttachedModuleConfig(
                name="SegmentationVisualizer",
                alias="segmentation_visualizer",
                attached_to="segmentation_head",
                params=self.visualizer_params,
            )
        ]
