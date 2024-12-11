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


class DetectionVariant(BaseModel):
    backbone: str
    backbone_params: Params
    neck_params: Params


def get_variant(variant: VariantLiteral) -> DetectionVariant:
    """Returns the specific variant configuration for the
    DetectionModel."""
    variants = {
        "light": DetectionVariant(
            backbone="EfficientRep",
            backbone_params={"variant": "n"},
            neck_params={"variant": "n"},
        ),
        "medium": DetectionVariant(
            backbone="EfficientRep",
            backbone_params={"variant": "s"},
            neck_params={"variant": "s"},
        ),
        "heavy": DetectionVariant(
            backbone="EfficientRep",
            backbone_params={"variant": "l"},
            neck_params={"variant": "l"},
        ),
    }

    if variant not in variants:
        raise ValueError(
            f"Detection variant should be one of {list(variants.keys())}, got '{variant}'."
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
        task_name: str | None = None,
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
        self.loss_params = loss_params or {"n_warmup_epochs": 0}
        self.visualizer_params = visualizer_params or {}
        self.task_name = task_name or "instance_segmentation"

    @property
    def nodes(self) -> list[ModelNodeConfig]:
        """Defines the model nodes, including backbone, neck, and
        head."""
        nodes = [
            ModelNodeConfig(
                name=self.backbone,
                alias=f"{self.backbone}-{self.task_name}",
                freezing=self.backbone_params.pop("freezing", {}),
                params=self.backbone_params,
            ),
        ]
        if self.use_neck:
            nodes.append(
                ModelNodeConfig(
                    name="RepPANNeck",
                    alias=f"RepPANNeck-{self.task_name}",
                    inputs=[f"{self.backbone}-{self.task_name}"],
                    freezing=self.neck_params.pop("freezing", {}),
                    params=self.neck_params,
                )
            )

        nodes.append(
            ModelNodeConfig(
                name="PrecisionSegmentBBoxHead",
                alias=f"PrecisionSegmentBBoxHead-{self.task_name}",
                freezing=self.head_params.pop("freezing", {}),
                inputs=[f"RepPANNeck-{self.task_name}"]
                if self.use_neck
                else [f"{self.backbone}-{self.task_name}"],
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
                name="PrecisionDFLSegmentationLoss",
                alias=f"PrecisionDFLSegmentationLoss-{self.task_name}",
                attached_to=f"PrecisionSegmentBBoxHead-{self.task_name}",
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
                alias=f"MeanAveragePrecision-{self.task_name}",
                attached_to=f"PrecisionSegmentBBoxHead-{self.task_name}",
                is_main_metric=True,
            ),
        ]

    @property
    def visualizers(self) -> list[AttachedModuleConfig]:
        """Defines the visualizer used for the detection task."""
        return [
            AttachedModuleConfig(
                name="InstanceSegmentationVisualizer",
                alias=f"InstanceSegmentationVisualizer-{self.task_name}",
                attached_to=f"PrecisionSegmentBBoxHead-{self.task_name}",
                params=self.visualizer_params,
            )
        ]
