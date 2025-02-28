from typing import Literal, TypeAlias

from luxonis_ml.typing import Kwargs
from pydantic import BaseModel

from luxonis_train.config import (
    AttachedModuleConfig,
    LossModuleConfig,
    MetricModuleConfig,
    ModelNodeConfig,
)

from .base_predefined_model import BasePredefinedModel

VariantLiteral: TypeAlias = Literal["light", "heavy"]


class FOMOVariant(BaseModel):
    backbone: str
    head_params: Kwargs
    backbone_params: Kwargs


def get_variant(variant: VariantLiteral) -> FOMOVariant:
    """Returns the specific variant configuration for the FOMOModel."""
    variants = {
        "light": FOMOVariant(
            backbone="EfficientRep",
            head_params={"num_conv_layers": 2, "conv_channels": 16},
            backbone_params={"variant": "n"},
        ),
        "heavy": FOMOVariant(
            backbone="MobileNetV2",
            head_params={"num_conv_layers": 2, "conv_channels": 16},
            backbone_params={},
        ),
    }

    if variant not in variants:
        raise ValueError(
            f"FOMO variant should be one of {list(variants.keys())}, got '{variant}'."
        )

    return variants[variant]


class FOMOModel(BasePredefinedModel):
    def __init__(
        self,
        variant: VariantLiteral = "light",
        backbone: str | None = None,
        backbone_params: Kwargs | None = None,
        head_params: Kwargs | None = None,
        loss_params: Kwargs | None = None,
        visualizer_params: Kwargs | None = None,
        task_name: str = "",
    ):
        var_config = get_variant(variant)

        self.backbone = backbone or var_config.backbone
        self.backbone_params = backbone_params or var_config.backbone_params
        self.head_params = head_params or var_config.head_params
        self.loss_params = loss_params or {}
        self.visualizer_params = visualizer_params or {}
        self.task_name = task_name

    @property
    def nodes(self) -> list[ModelNodeConfig]:
        nodes = [
            ModelNodeConfig(
                name=self.backbone,
                alias=f"{self.task_name}-{self.backbone}",
                freezing=self.backbone_params.pop("freezing", {}),
                params=self.backbone_params,
            ),
            ModelNodeConfig(
                name="FOMOHead",
                alias=f"{self.task_name}-FOMOHead",
                inputs=[f"{self.task_name}-{self.backbone}"],
                params=self.head_params,
                task_name=self.task_name,
            ),
        ]
        return nodes

    @property
    def losses(self) -> list[LossModuleConfig]:
        return [
            LossModuleConfig(
                name="FOMOLocalizationLoss",
                attached_to=f"{self.task_name}-FOMOHead",
                params=self.loss_params,
                weight=1.0,
            )
        ]

    @property
    def metrics(self) -> list[MetricModuleConfig]:
        return [
            MetricModuleConfig(
                name="ObjectKeypointSimilarity",
                attached_to=f"{self.task_name}-FOMOHead",
                is_main_metric=True,
            ),
        ]

    @property
    def visualizers(self) -> list[AttachedModuleConfig]:
        return [
            AttachedModuleConfig(
                name="FOMOVisualizer",
                attached_to=f"{self.task_name}-FOMOHead",
                params=self.visualizer_params,
            )
        ]
