from typing import Literal, TypeAlias

from pydantic import BaseModel

from luxonis_train.config import (
    AttachedModuleConfig,
    LossModuleConfig,
    MetricModuleConfig,
    ModelNodeConfig,
    Params,
)

from .base_predefined_model import BasePredefinedModel

VariantLiteral: TypeAlias = Literal["light", "heavy"]


class ClassificationVariant(BaseModel):
    backbone: str
    backbone_params: Params


def get_variant(variant: VariantLiteral) -> ClassificationVariant:
    """Returns the specific variant configuration for the
    ClassificationModel."""
    variants = {
        "light": ClassificationVariant(
            backbone="ResNet",
            backbone_params={"variant": "18"},
        ),
        "heavy": ClassificationVariant(
            backbone="ResNet",
            backbone_params={"variant": "101"},
        ),
    }

    if variant not in variants:
        raise ValueError(
            f"Classification variant should be one of {list(variants.keys())}, got '{variant}'."
        )

    return variants[variant]


class ClassificationModel(BasePredefinedModel):
    def __init__(
        self,
        variant: VariantLiteral = "light",
        backbone: str | None = None,
        backbone_params: Params | None = None,
        head_params: Params | None = None,
        loss_params: Params | None = None,
        visualizer_params: Params | None = None,
        task: Literal["multiclass", "multilabel"] = "multiclass",
        task_name: str | None = None,
    ):
        var_config = get_variant(variant)

        self.backbone = backbone or var_config.backbone
        self.backbone_params = (
            backbone_params
            if backbone is not None or backbone_params is not None
            else var_config.backbone_params
        ) or {}
        self.head_params = head_params or {}
        self.loss_params = loss_params or {}
        self.visualizer_params = visualizer_params or {}
        self.task = task
        self.task_name = task_name or "classification"

    @property
    def nodes(self) -> list[ModelNodeConfig]:
        """Defines the model nodes, including backbone and head."""
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
                task=self.task_name,
            ),
        ]

    @property
    def losses(self) -> list[LossModuleConfig]:
        """Defines the loss module for the classification task."""
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
        """Defines the metrics used for evaluation."""
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
        """Defines the visualizer used for the classification task."""
        return [
            AttachedModuleConfig(
                name="ClassificationVisualizer",
                attached_to="classification_head",
                params=self.visualizer_params,
            )
        ]
