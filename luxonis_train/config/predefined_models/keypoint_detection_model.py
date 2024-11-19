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
    neck_params: Params


def get_variant(variant: VariantLiteral) -> KeypointDetectionVariant:
    """Returns the specific variant configuration for the
    KeypointDetectionModel."""
    variants = {
        "light": KeypointDetectionVariant(
            backbone="EfficientRep",
            backbone_params={"variant": "n"},
            neck_params={"variant": "n", "download_weights": True},
        ),
        "medium": KeypointDetectionVariant(
            backbone="EfficientRep",
            backbone_params={"variant": "s"},
            neck_params={"variant": "s", "download_weights": True},
        ),
        "heavy": KeypointDetectionVariant(
            backbone="EfficientRep",
            backbone_params={"variant": "l"},
            neck_params={"variant": "l", "download_weights": True},
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
        backbone: str | None = None,
        backbone_params: Params | None = None,
        neck_params: Params | None = None,
        head_params: Params | None = None,
        loss_params: Params | None = None,
        kpt_visualizer_params: Params | None = None,
        bbox_visualizer_params: Params | None = None,
        bbox_task_name: str | None = None,
        kpt_task_name: str | None = None,
    ):
        var_config = get_variant(variant)

        self.use_neck = use_neck
        self.backbone = backbone or var_config.backbone
        self.backbone_params = (
            backbone_params
            if backbone is not None or backbone_params is not None
            else var_config.backbone_params
        ) or {}
        self.neck_params = neck_params or var_config.neck_params
        self.head_params = head_params or {}
        self.loss_params = loss_params or {"n_warmup_epochs": 0}
        self.kpt_visualizer_params = kpt_visualizer_params or {}
        self.bbox_visualizer_params = bbox_visualizer_params or {}
        self.bbox_task_name = bbox_task_name or "boundingbox"
        self.kpt_task_name = kpt_task_name or "keypoints"

    @property
    def nodes(self) -> list[ModelNodeConfig]:
        """Defines the model nodes, including backbone, neck, and
        head."""
        nodes = [
            ModelNodeConfig(
                name=self.backbone,
                alias=f"{self.backbone}-{self.kpt_task_name}",
                freezing=self.backbone_params.pop("freezing", {}),
                params=self.backbone_params,
            ),
        ]
        if self.use_neck:
            nodes.append(
                ModelNodeConfig(
                    name="RepPANNeck",
                    alias=f"RepPANNeck-{self.kpt_task_name}",
                    inputs=[f"{self.backbone}-{self.kpt_task_name}"],
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
                name="EfficientKeypointBBoxHead",
                alias=f"EfficientKeypointBBoxHead-{self.kpt_task_name}",
                inputs=(
                    [f"RepPANNeck-{self.kpt_task_name}"]
                    if self.use_neck
                    else [f"{self.backbone}-{self.kpt_task_name}"]
                ),
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
                name="EfficientKeypointBBoxLoss",
                alias=f"EfficientKeypointBBoxLoss-{self.kpt_task_name}",
                attached_to=f"EfficientKeypointBBoxHead-{self.kpt_task_name}",
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
                alias=f"ObjectKeypointSimilarity-{self.kpt_task_name}",
                attached_to=f"EfficientKeypointBBoxHead-{self.kpt_task_name}",
                is_main_metric=True,
            ),
            MetricModuleConfig(
                name="MeanAveragePrecisionKeypoints",
                alias=f"MeanAveragePrecisionKeypoints-{self.kpt_task_name}",
                attached_to=f"EfficientKeypointBBoxHead-{self.kpt_task_name}",
            ),
        ]

    @property
    def visualizers(self) -> list[AttachedModuleConfig]:
        """Defines the visualizer used for the keypoint detection
        task."""
        return [
            AttachedModuleConfig(
                name="MultiVisualizer",
                alias=f"MultiVisualizer-{self.kpt_task_name}",
                attached_to=f"EfficientKeypointBBoxHead-{self.kpt_task_name}",
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
