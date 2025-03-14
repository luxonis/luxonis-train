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


class InstanceSegmentationVariant(BaseModel):
    backbone: str
    backbone_params: Params
    neck_params: Params


def get_variant(variant: VariantLiteral) -> InstanceSegmentationVariant:
    """Returns the specific variant configuration for the
    InstanceSegmentationModel."""
    variants = {
        "light": InstanceSegmentationVariant(
            backbone="EfficientRep",
            backbone_params={"variant": "n"},
            neck_params={"variant": "n"},
        ),
        "medium": InstanceSegmentationVariant(
            backbone="EfficientRep",
            backbone_params={"variant": "s"},
            neck_params={"variant": "s"},
        ),
        "heavy": InstanceSegmentationVariant(
            backbone="EfficientRep",
            backbone_params={"variant": "l"},
            neck_params={"variant": "l"},
        ),
    }

    if variant not in variants:
        raise ValueError(
            f"Instance segmentation variant should be one of {list(variants.keys())}, got '{variant}'."
        )

    return variants[variant]


class InstanceSegmentationModel(BasePredefinedModel):
    def __init__(
        self,
        variant: VariantLiteral = "light",
        use_neck: bool = True,
        backbone: str | None = None,
        backbone_params: Params | None = None,
        neck_params: Params | None = None,
        head_params: Params | None = None,
        loss_params: Params | None = None,
        visualizer_params: Params | None = None,
        task_name: str = "",
        enable_confusion_matrix: bool = True,
        confusion_matrix_params: Params | None = None,
    ):
        var_config = get_variant(variant)

        self.use_neck = use_neck
        self.backbone_params = (
            backbone_params
            if backbone is not None or backbone_params is not None
            else var_config.backbone_params
        ) or {}
        self.backbone = backbone or var_config.backbone
        self.neck_params = neck_params or var_config.neck_params
        self.head_params = head_params or {}
        self.loss_params = loss_params or {}
        self.visualizer_params = visualizer_params or {}
        self.task_name = task_name
        self.enable_confusion_matrix = enable_confusion_matrix
        self.confusion_matrix_params = confusion_matrix_params or {}

    @property
    def nodes(self) -> list[ModelNodeConfig]:
        """Defines the model nodes, including backbone, neck, and
        head."""
        nodes = [
            ModelNodeConfig(
                name=self.backbone,
                freezing=self.backbone_params.pop("freezing", {}),
                params=self.backbone_params,
            ),
        ]
        if self.use_neck:
            nodes.append(
                ModelNodeConfig(
                    name="RepPANNeck",
                    inputs=[f"{self.backbone}"],
                    freezing=self.neck_params.pop("freezing", {}),
                    params=self.neck_params,
                )
            )

        nodes.append(
            ModelNodeConfig(
                name="PrecisionSegmentBBoxHead",
                freezing=self.head_params.pop("freezing", {}),
                inputs=["RepPANNeck"]
                if self.use_neck
                else [f"{self.backbone}"],
                params=self.head_params,
                task_name=self.task_name,
            )
        )
        return nodes

    @property
    def losses(self) -> list[LossModuleConfig]:
        """Defines the loss module for the instance segmentation
        task."""
        return [
            LossModuleConfig(
                name="PrecisionDFLSegmentationLoss",
                attached_to="PrecisionSegmentBBoxHead",
                params=self.loss_params,
                weight=1.0,
            )
        ]

    @property
    def metrics(self) -> list[MetricModuleConfig]:
        """Defines the metrics used for evaluation."""
        metrics = [
            MetricModuleConfig(
                name="MeanAveragePrecision",
                attached_to="PrecisionSegmentBBoxHead",
                is_main_metric=True,
            ),
        ]
        if self.enable_confusion_matrix:
            metrics.append(
                MetricModuleConfig(
                    name="ConfusionMatrix",
                    alias="ConfusionMatrix",
                    attached_to="PrecisionSegmentBBoxHead",
                    params={**self.confusion_matrix_params},
                )
            )
        return metrics

    @property
    def visualizers(self) -> list[AttachedModuleConfig]:
        """Defines the visualizer used for the instance segmentation
        task."""
        return [
            AttachedModuleConfig(
                name="InstanceSegmentationVisualizer",
                attached_to="PrecisionSegmentBBoxHead",
                params=self.visualizer_params,
            )
        ]
