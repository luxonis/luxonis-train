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


class KeypointDetectionVariant(BaseModel):
    backbone: str
    backbone_params: Params


def get_variant(variant: VariantLiteral) -> KeypointDetectionVariant:
    """Returns the specific variant configuration for the
    KeypointDetectionModel."""
    variants = {
        "light": KeypointDetectionVariant(
            backbone="EfficientRep",
            backbone_params={"variant": "n"},
        ),
        "heavy": KeypointDetectionVariant(
            backbone="EfficientRep",
            backbone_params={"variant": "l"},
        ),
    }

    if variant not in variants:
        raise ValueError(
            f"KeypointDetection variant should be one of {list(variants.keys())}, got '{variant}'."
        )

    return variants[variant]


class KeypointDetectionModel(BasePredefinedModel):
    def __init__(
        self,
        variant: VariantLiteral = "light",
        use_neck: bool = True,
        backbone_params: Params | None = None,
        neck_params: Params | None = None,
        head_params: Params | None = None,
        loss_params: Params | None = None,
        head_type: Literal[
            "ImplicitKeypointBBoxHead", "EfficientKeypointBBoxHead"
        ] = "EfficientKeypointBBoxHead",
        kpt_visualizer_params: Params | None = None,
        bbox_visualizer_params: Params | None = None,
        bbox_task_name: str | None = None,
        kpt_task_name: str | None = None,
        backbone: str | None = None,
    ):
        self.variant = variant
        self.use_neck = use_neck

        var_config = get_variant(variant)

        self.backbone_params = backbone_params or var_config.backbone_params
        self.neck_params = neck_params or {}
        self.head_params = head_params or {}
        self.loss_params = loss_params or {"n_warmup_epochs": 0}
        self.kpt_visualizer_params = kpt_visualizer_params or {}
        self.bbox_visualizer_params = bbox_visualizer_params or {}
        self.bbox_task_name = bbox_task_name
        self.kpt_task_name = kpt_task_name
        self.head_type = head_type
        self.backbone = backbone or var_config.backbone

    @property
    def nodes(self) -> list[ModelNodeConfig]:
        """Defines the model nodes, including backbone, neck, and
        head."""
        nodes = [
            ModelNodeConfig(
                name=self.backbone,
                alias="kpt_detection_backbone",
                freezing=self.backbone_params.pop("freezing", {}),
                params=self.backbone_params,
            ),
        ]
        if self.use_neck:
            nodes.append(
                ModelNodeConfig(
                    name="RepPANNeck",
                    alias="kpt_detection_neck",
                    inputs=["kpt_detection_backbone"],
                    freezing=self.neck_params.pop("freezing", {}),
                    params=self.neck_params,
                )
            )

        task = {}
        if self.bbox_task_name is not None:
            task["boundingbox"] = self.bbox_task_name
        if self.kpt_task_name is not None:
            task["keypoints"] = self.kpt_task_name

        nodes.append(
            ModelNodeConfig(
                name=self.head_type,
                alias="kpt_detection_head",
                inputs=["kpt_detection_neck"]
                if self.use_neck
                else ["kpt_detection_backbone"],
                freezing=self.head_params.pop("freezing", {}),
                params=self.head_params,
                task=task,
            )
        )
        return nodes

    @property
    def losses(self) -> list[LossModuleConfig]:
        """Defines the loss module for the keypoint detection task."""
        return [
            LossModuleConfig(
                name=self.head_type.replace("Head", "Loss"),
                attached_to="kpt_detection_head",
                params=self.loss_params,
                weight=1.0,
            )
        ]

    @property
    def metrics(self) -> list[MetricModuleConfig]:
        """Defines the metrics used for evaluation."""
        return [
            MetricModuleConfig(
                name="ObjectKeypointSimilarity",
                alias="kpt_detection_oks",
                attached_to="kpt_detection_head",
                is_main_metric=True,
            ),
            MetricModuleConfig(
                name="MeanAveragePrecisionKeypoints",
                alias="kpt_detection_map",
                attached_to="kpt_detection_head",
            ),
        ]

    @property
    def visualizers(self) -> list[AttachedModuleConfig]:
        """Defines the visualizer used for the keypoint detection
        task."""
        return [
            AttachedModuleConfig(
                name="MultiVisualizer",
                alias="kpt_detection_visualizer",
                attached_to="kpt_detection_head",
                params={
                    "visualizers": [
                        {
                            "name": "KeypointVisualizer",
                            "params": self.kpt_visualizer_params,
                        },
                        {
                            "name": "BBoxVisualizer",
                            "params": self.bbox_visualizer_params,
                        },
                    ]
                },
            )
        ]
