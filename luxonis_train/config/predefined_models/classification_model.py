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
            backbone_params={"variant": "50"},
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
                alias=f"{self.backbone}-{self.task_name}",
                freezing=self.backbone_params.pop("freezing", {}),
                params=self.backbone_params,
            ),
            ModelNodeConfig(
                name="ClassificationHead",
                alias=f"ClassificationHead-{self.task_name}",
                inputs=[f"{self.backbone}-{self.task_name}"],
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
                alias=f"CrossEntropyLoss-{self.task_name}",
                attached_to=f"ClassificationHead-{self.task_name}",
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
                alias=f"F1Score-{self.task_name}",
                is_main_metric=True,
                attached_to=f"ClassificationHead-{self.task_name}",
                params={"task": self.task},
            ),
            MetricModuleConfig(
                name="Accuracy",
                alias=f"Accuracy-{self.task_name}",
                attached_to=f"ClassificationHead-{self.task_name}",
                params={"task": self.task},
            ),
            MetricModuleConfig(
                name="Recall",
                alias=f"Recall-{self.task_name}",
                attached_to=f"ClassificationHead-{self.task_name}",
                params={"task": self.task},
            ),
        ]

    @property
    def visualizers(self) -> list[AttachedModuleConfig]:
        """Defines the visualizer used for the classification task."""
        return [
            AttachedModuleConfig(
                name="ClassificationVisualizer",
                alias=f"ClassificationVisualizer-{self.task_name}",
                attached_to=f"ClassificationHead-{self.task_name}",
                params=self.visualizer_params,
            )
        ]
