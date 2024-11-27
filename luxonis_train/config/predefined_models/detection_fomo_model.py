from typing import Literal, TypeAlias

from pydantic import BaseModel

from luxonis_train.config import (
    AttachedModuleConfig,
    LossModuleConfig,
    MetricModuleConfig,
    ModelNodeConfig,
    Params,
)
from luxonis_train.enums import TaskType

from .base_predefined_model import BasePredefinedModel

VariantLiteral: TypeAlias = Literal["light", "heavy"]


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
        kpt_visualizer_params: Params | None = None,
        bbox_task_name: str | None = None,
        kpt_task_name: str | None = None,
    ):
        var_config = get_variant(variant)

        self.backbone = backbone or var_config.backbone
        self.backbone_params = backbone_params or {}
        self.head_params = head_params or var_config.head_params
        self.loss_params = loss_params or {}
        self.kpt_visualizer_params = kpt_visualizer_params or {}
        self.bbox_task_name = (
            bbox_task_name or "boundingbox"
        )  # Needed for OKS calculation
        self.kpt_task_name = kpt_task_name or "keypoints"

    @property
    def nodes(self) -> list[ModelNodeConfig]:
        nodes = [
            ModelNodeConfig(
                name=self.backbone,
                alias=f"{self.backbone}-{self.kpt_task_name}",
                freezing=self.backbone_params.pop("freezing", {}),
                params=self.backbone_params,
            ),
            ModelNodeConfig(
                name="FOMOHead",
                alias=f"FOMOHead-{self.kpt_task_name}",
                inputs=[f"{self.backbone}-{self.kpt_task_name}"],
                params=self.head_params,
                task={
                    TaskType.BOUNDINGBOX: self.bbox_task_name,
                    TaskType.KEYPOINTS: self.kpt_task_name,
                },
            ),
        ]
        return nodes

    @property
    def losses(self) -> list[LossModuleConfig]:
        return [
            LossModuleConfig(
                name="FOMOLocalizationLoss",
                alias=f"FOMOLocalizationLoss-{self.kpt_task_name}",
                attached_to=f"FOMOHead-{self.kpt_task_name}",
                params=self.loss_params,
                weight=1.0,
            )
        ]

    @property
    def metrics(self) -> list[MetricModuleConfig]:
        return [
            MetricModuleConfig(
                name="ObjectKeypointSimilarity",
                alias=f"ObjectKeypointSimilarity-{self.kpt_task_name}",
                attached_to=f"FOMOHead-{self.kpt_task_name}",
                is_main_metric=True,
            ),
        ]

    @property
    def visualizers(self) -> list[AttachedModuleConfig]:
        return [
            AttachedModuleConfig(
                name="MultiVisualizer",
                alias=f"MultiVisualizer-{self.kpt_task_name}",
                attached_to=f"FOMOHead-{self.kpt_task_name}",
                params={
                    "visualizers": [
                        {
                            "name": "KeypointVisualizer",
                            "params": self.kpt_visualizer_params,
                        },
                    ]
                },
            )
        ]
