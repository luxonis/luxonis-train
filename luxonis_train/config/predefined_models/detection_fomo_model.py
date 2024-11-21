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

VariantLiteral: TypeAlias = Literal["light", "medium", "heavy"]


class FOMOVariant(BaseModel):
    backbone: str
    head_params: Params


def get_variant(variant: VariantLiteral) -> FOMOVariant:
    """Returns the specific variant configuration for the FOMOModel."""
    variants = {
        "light": FOMOVariant(
            backbone="MobileNetV2",
            head_params={"num_conv_layers": 2, "conv_channels": 16},
        ),
        "heavy": FOMOVariant(
            backbone="MobileNetV2",
            head_params={"num_conv_layers": 5, "conv_channels": 64},
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
        backbone_params: Params | None = None,
        head_params: Params | None = None,
        loss_params: Params | None = None,
        visualizer_params: Params | None = None,
        task_name: str | None = None,
    ):
        var_config = get_variant(variant)

        self.backbone = backbone or var_config.backbone
        self.backbone_params = backbone_params or {}
        self.head_params = head_params or var_config.head_params
        self.loss_params = loss_params or {}
        self.visualizer_params = visualizer_params or {}
        self.task_name = task_name or "keypoints"

    @property
    def nodes(self) -> list[ModelNodeConfig]:
        nodes = [
            ModelNodeConfig(
                name=self.backbone,
                alias=f"{self.backbone}-{self.task_name}",
                freezing=self.backbone_params.pop("freezing", {}),
                params=self.backbone_params,
            ),
            ModelNodeConfig(
                name="FOMOHead",
                alias=f"FOMOHead-{self.task_name}",
                inputs=[f"{self.backbone}-{self.task_name}"],
                params=self.head_params,
                task=self.task_name,
            ),
        ]
        return nodes

    @property
    def losses(self) -> list[LossModuleConfig]:
        return [
            LossModuleConfig(
                name="FOMOLocalizationLoss",
                alias=f"FOMOLocalizationLoss-{self.task_name}",
                attached_to=f"FOMOHead-{self.task_name}",
                params=self.loss_params,
                weight=1.0,
            )
        ]

    @property
    def metrics(self) -> list[MetricModuleConfig]:
        return [
            MetricModuleConfig(
                name="ObjectKeypointSimilarity",
                alias=f"ObjectKeypointSimilarity-{self.task_name}",
                attached_to=f"FOMOHead-{self.task_name}",
                is_main_metric=True,
            ),
        ]

    @property
    def visualizers(self) -> list[AttachedModuleConfig]:
        return [
            AttachedModuleConfig(
                name="KeypointVisualizer",
                alias=f"KeypointVisualizer-{self.task_name}",
                attached_to=f"FOMOHead-{self.task_name}",
                params=self.visualizer_params,
            )
        ]
