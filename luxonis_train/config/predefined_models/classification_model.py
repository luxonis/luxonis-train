from typing import Literal, TypeAlias

from luxonis_ml.typing import Params
from pydantic import BaseModel

from luxonis_train.config import (
    AttachedModuleConfig,
    LossModuleConfig,
    MetricModuleConfig,
    NodeConfig,
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
            backbone="ResNet", backbone_params={"variant": "18"}
        ),
        "heavy": ClassificationVariant(
            backbone="ResNet", backbone_params={"variant": "50"}
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
        task_name: str = "",
        enable_confusion_matrix: bool = True,
        confusion_matrix_params: Params | None = None,
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
        self.task_name = task_name
        self.enable_confusion_matrix = enable_confusion_matrix
        self.confusion_matrix_params = confusion_matrix_params or {}

    @property
    def nodes(self) -> list[NodeConfig]:
        """Defines the model nodes, including backbone and head."""
        return [
            NodeConfig(
                name=self.backbone,
                freezing=self._get_freezing(self.backbone_params),
                params=self.backbone_params,
            ),
            NodeConfig(
                name="ClassificationHead",
                freezing=self._get_freezing(self.head_params),
                inputs=[self.backbone],
                params=self.head_params,
                task_name=self.task_name,
            ),
        ]

    @property
    def losses(self) -> list[LossModuleConfig]:
        """Defines the loss module for the classification task."""
        return [
            LossModuleConfig(
                name="CrossEntropyLoss",
                attached_to="ClassificationHead",
                params=self.loss_params,
                weight=1.0,
            )
        ]

    @property
    def metrics(self) -> list[MetricModuleConfig]:
        """Defines the metrics used for evaluation."""
        metrics = [
            MetricModuleConfig(
                name="F1Score",
                is_main_metric=True,
                attached_to="ClassificationHead",
                params={"task": self.task},
            ),
            MetricModuleConfig(
                name="Accuracy",
                attached_to="ClassificationHead",
                params={"task": self.task},
            ),
            MetricModuleConfig(
                name="Recall",
                attached_to="ClassificationHead",
                params={"task": self.task},
            ),
        ]
        if self.enable_confusion_matrix:
            metrics.append(
                MetricModuleConfig(
                    name="ConfusionMatrix",
                    attached_to="ClassificationHead",
                    params={**self.confusion_matrix_params},
                )
            )
        return metrics

    @property
    def visualizers(self) -> list[AttachedModuleConfig]:
        """Defines the visualizer used for the classification task."""
        return [
            AttachedModuleConfig(
                name="ClassificationVisualizer",
                attached_to="ClassificationHead",
                params=self.visualizer_params,
            )
        ]
