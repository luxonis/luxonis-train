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
class ClassificationModel(BasePredefinedModel):
    backbone: str = "MicroNet"
    task: Literal["multiclass", "multilabel"] = "multilabel"
    backbone_params: Kwargs = field(default_factory=dict)
    head_params: Kwargs = field(default_factory=dict)
    loss_params: Kwargs = field(default_factory=dict)
    visualizer_params: Kwargs = field(default_factory=dict)

    @property
    def nodes(self) -> list[ModelNodeConfig]:
        return [
            ModelNodeConfig(
                name=self.backbone,
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
