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


class DetectionVariant(BaseModel):
    backbone: str
    backbone_params: Params


def get_variant(variant: VariantLiteral) -> DetectionVariant:
    """Returns the specific variant configuration for the
    DetectionModel."""
    variants = {
        "light": DetectionVariant(
            backbone="EfficientRep",
            backbone_params={"variant": "n"},
        ),
        "heavy": DetectionVariant(
            backbone="EfficientRep",
            backbone_params={"variant": "l"},
        ),
    }

    if variant not in variants:
        raise ValueError(
            f"Detection variant should be one of {list(variants.keys())}, got '{variant}'."
        )

    return variants[variant]


class DetectionModel(BasePredefinedModel):
    def __init__(
        self,
        variant: VariantLiteral = "light",
        use_neck: bool = True,
        backbone_params: Params | None = None,
        neck_params: Params | None = None,
        head_params: Params | None = None,
        loss_params: Params | None = None,
        task_name: str | None = None,
        visualizer_params: Params | None = None,
        backbone: str | None = None,
    ):
        self.variant = variant
        self.use_neck = use_neck

        var_config = get_variant(variant)

        self.backbone_params = backbone_params or var_config.backbone_params
        self.neck_params = neck_params or {}
        self.head_params = head_params or {}
        self.loss_params = loss_params or {"n_warmup_epochs": 0}
        self.visualizer_params = visualizer_params or {}
        self.task_name = task_name or "boundingbox"
        self.backbone = backbone or var_config.backbone

    @property
    def nodes(self) -> list[ModelNodeConfig]:
        """Defines the model nodes, including backbone, neck, and
        head."""
        nodes = [
            ModelNodeConfig(
                name=self.backbone,
                alias="detection_backbone",
                freezing=self.backbone_params.pop("freezing", {}),
                params=self.backbone_params,
            ),
        ]
        if self.use_neck:
            nodes.append(
                ModelNodeConfig(
                    name="RepPANNeck",
                    alias="detection_neck",
                    inputs=["detection_backbone"],
                    freezing=self.neck_params.pop("freezing", {}),
                    params=self.neck_params,
                )
            )

        nodes.append(
            ModelNodeConfig(
                name="EfficientBBoxHead",
                alias="detection_head",
                freezing=self.head_params.pop("freezing", {}),
                inputs=["detection_neck"]
                if self.use_neck
                else ["detection_backbone"],
                params=self.head_params,
                task=self.task_name,
            )
        )
        return nodes

    @property
    def losses(self) -> list[LossModuleConfig]:
        """Defines the loss module for the detection task."""
        return [
            LossModuleConfig(
                name="AdaptiveDetectionLoss",
                alias="detection_loss",
                attached_to="detection_head",
                params=self.loss_params,
                weight=1.0,
            )
        ]

    @property
    def metrics(self) -> list[MetricModuleConfig]:
        """Defines the metrics used for evaluation."""
        return [
            MetricModuleConfig(
                name="MeanAveragePrecision",
                alias="detection_map",
                attached_to="detection_head",
                is_main_metric=True,
            ),
        ]

    @property
    def visualizers(self) -> list[AttachedModuleConfig]:
        """Defines the visualizer used for the detection task."""
        return [
            AttachedModuleConfig(
                name="BBoxVisualizer",
                alias="detection_visualizer",
                attached_to="detection_head",
                params=self.visualizer_params,
            )
        ]
