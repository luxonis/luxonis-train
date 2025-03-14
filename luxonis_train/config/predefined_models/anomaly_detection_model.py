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


class AnomalyVariant(BaseModel):
    backbone: str
    backbone_params: Params
    head_params: Params


def get_variant(variant: VariantLiteral) -> AnomalyVariant:
    """Returns the specific variant configuration for the
    AnomalyDetectionModel."""
    variants = {
        "light": AnomalyVariant(
            backbone="RecSubNet",
            backbone_params={"variant": "n"},
            head_params={"variant": "n"},
        ),
        "heavy": AnomalyVariant(
            backbone="RecSubNet",
            backbone_params={"variant": "l"},
            head_params={"variant": "l"},
        ),
    }

    if variant not in variants:
        raise ValueError(
            f"Anomaly detection variant should be one of {list(variants.keys())}, got '{variant}'."
        )

    return variants[variant]


class AnomalyDetectionModel(BasePredefinedModel):
    def __init__(
        self,
        variant: VariantLiteral = "light",
        backbone: str | None = None,
        backbone_params: Params | None = None,
        loss_params: Params | None = None,
        visualizer_params: Params | None = None,
        head_params: Params | None = None,
        task_name: str = "",
    ):
        var_config = get_variant(variant)

        self.backbone = backbone or var_config.backbone
        self.backbone_params = (
            backbone_params
            if backbone is not None or backbone_params is not None
            else var_config.backbone_params
        ) or {}
        self.head_params = head_params or var_config.head_params
        self.loss_params = loss_params or {}
        self.visualizer_params = visualizer_params or {}
        self.task_name = task_name

    @property
    def nodes(self) -> list[ModelNodeConfig]:
        """Defines the model nodes, including RecSubNet and
        DiscSubNetHead."""
        return [
            ModelNodeConfig(
                name=self.backbone,
                params=self.backbone_params,
            ),
            ModelNodeConfig(
                name="DiscSubNetHead",
                inputs=[f"{self.backbone}"],
                params=self.head_params,
                task_name=self.task_name,
            ),
        ]

    @property
    def losses(self) -> list[LossModuleConfig]:
        """Defines the loss module for the anomaly detection task."""
        return [
            LossModuleConfig(
                name="ReconstructionSegmentationLoss",
                attached_to="DiscSubNetHead",
                params=self.loss_params,
                weight=1.0,
            )
        ]

    @property
    def metrics(self) -> list[MetricModuleConfig]:
        """Defines the metrics used for evaluation."""
        return [
            MetricModuleConfig(
                name="JaccardIndex",
                attached_to="DiscSubNetHead",
                params={"num_classes": 2, "task": "multiclass"},
                is_main_metric=True,
            ),
        ]

    @property
    def visualizers(self) -> list[AttachedModuleConfig]:
        """Defines the visualizer used for the anomaly detection
        task."""
        return [
            AttachedModuleConfig(
                name="SegmentationVisualizer",
                attached_to="DiscSubNetHead",
                params=self.visualizer_params,
            )
        ]
