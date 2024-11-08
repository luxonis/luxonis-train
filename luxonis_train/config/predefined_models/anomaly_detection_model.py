from typing import Literal, TypeAlias

from pydantic import BaseModel

from luxonis_train.config import (
    AttachedModuleConfig,
    LossModuleConfig,
    MetricModuleConfig,  # Metrics support added
    ModelNodeConfig,
    Params,
)

from .base_predefined_model import BasePredefinedModel

VariantLiteral: TypeAlias = Literal["light", "heavy"]


class AnomalyVariant(BaseModel):
    backbone: str
    backbone_params: Params


def get_variant(variant: VariantLiteral) -> AnomalyVariant:
    """Returns the specific variant configuration for the
    AnomalyDetectionModel."""
    variants = {
        "light": AnomalyVariant(
            backbone="RecSubNet",
            backbone_params={"variant": "n"},
        ),
        "heavy": AnomalyVariant(
            backbone="RecSubNet",
            backbone_params={"variant": "l"},
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
        disc_subnet_params: Params | None = None,
        loss_params: Params | None = None,
        visualizer_params: Params | None = None,
        task_name: str | None = None,
    ):
        var_config = get_variant(variant)

        self.backbone = backbone or var_config.backbone
        self.backbone_params = (
            backbone_params
            if backbone is not None or backbone_params is not None
            else var_config.backbone_params
        ) or {}
        self.disc_subnet_params = disc_subnet_params or {}
        self.loss_params = loss_params or {}
        self.visualizer_params = visualizer_params or {}
        self.task_name = task_name or "anomaly_detection"

    @property
    def nodes(self) -> list[ModelNodeConfig]:
        """Defines the model nodes, including RecSubNet and
        DiscSubNetHead."""
        return [
            ModelNodeConfig(
                name=self.backbone,
                alias=f"{self.backbone}-{self.task_name}",
                params=self.backbone_params,
            ),
            ModelNodeConfig(
                name="DiscSubNetHead",
                alias=f"DiscSubNetHead-{self.task_name}",
                inputs=[f"{self.backbone}-{self.task_name}"],
                params=self.disc_subnet_params,
            ),
        ]

    @property
    def losses(self) -> list[LossModuleConfig]:
        """Defines the loss module for the anomaly detection task."""
        return [
            LossModuleConfig(
                name="ReconstructionSegmentationLoss",
                alias=f"ReconstructionSegmentationLoss-{self.task_name}",
                attached_to=f"DiscSubNetHead-{self.task_name}",
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
                alias=f"JaccardIndex-{self.task_name}",
                attached_to=f"DiscSubNetHead-{self.task_name}",
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
                alias=f"SegmentationVisualizer-{self.task_name}",
                attached_to=f"DiscSubNetHead-{self.task_name}",
                params=self.visualizer_params,
            )
        ]
