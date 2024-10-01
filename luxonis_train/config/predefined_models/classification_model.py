from dataclasses import dataclass, field
from typing import Literal

from luxonis_train.config import (
    AttachedModuleConfig,
    LossModuleConfig,
    MetricModuleConfig,
    ModelNodeConfig,
    Params,
)

from .base_predefined_model import BasePredefinedModel


@dataclass
class ClassificationModel(BasePredefinedModel):
    variant: str | None = None
    task: Literal["multiclass", "multilabel"] = "multiclass"
    backbone_params: Params = field(default_factory=dict)
    head_params: Params = field(default_factory=dict)
    loss_params: Params = field(default_factory=dict)
    task_name: str | None = None
    visualizer_params: Params = field(
        default_factory=lambda: {
            "font_scale": 0.5,
            "color": [255, 0, 0],
            "thickness": 2,
            "include_plot": True,
        }
    )

    def __post_init__(self):
        if self.variant == "heavy":
            self.backbone_params.setdefault("variant", "101")
        elif self.variant == "light":
            self.backbone_params.setdefault("variant", "18")

    @property
    def nodes(self) -> list[ModelNodeConfig]:
        return [
            ModelNodeConfig(
                name="ResNet",
                alias="classification_backbone",
                freezing=self.backbone_params.pop("freezing", {}),
                params=self.backbone_params,
            ),
            ModelNodeConfig(
                name="ClassificationHead",
                alias="classification_head",
                inputs=["classification_backbone"],
                freezing=self.head_params.pop("freezing", {}),
                params=self.head_params,
                task=self.task_name,
            ),
        ]

    @property
    def losses(self) -> list[LossModuleConfig]:
        return [
            LossModuleConfig(
                name="CrossEntropyLoss",
                alias="classification_loss",
                attached_to="classification_head",
                params=self.loss_params,
                weight=1.0,
            )
        ]

    @property
    def metrics(self) -> list[MetricModuleConfig]:
        return [
            MetricModuleConfig(
                name="F1Score",
                alias="classification_f1_score",
                is_main_metric=True,
                attached_to="classification_head",
                params={"task": self.task},
            ),
            MetricModuleConfig(
                name="Accuracy",
                alias="classification_accuracy",
                attached_to="classification_head",
                params={"task": self.task},
            ),
            MetricModuleConfig(
                name="Recall",
                alias="classification_recall",
                attached_to="classification_head",
                params={"task": self.task},
            ),
        ]

    @property
    def visualizers(self) -> list[AttachedModuleConfig]:
        return [
            AttachedModuleConfig(
                name="ClassificationVisualizer",
                attached_to="classification_head",
                params=self.visualizer_params,
            )
        ]
