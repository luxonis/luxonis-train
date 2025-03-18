from typing import Literal, TypeAlias

from luxonis_ml.typing import Params
from pydantic import BaseModel

from luxonis_train.config import (
    AttachedModuleConfig,
    LossModuleConfig,
    MetricModuleConfig,
    NodeConfig,
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
        visualizer_params: Params | None = None,
        task_name: str = "",
        enable_confusion_matrix: bool = True,
        confusion_matrix_params: Params | None = None,
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
        self.visualizer_params = visualizer_params or {}
        self.task_name = task_name
        self.enable_confusion_matrix = enable_confusion_matrix
        self.confusion_matrix_params = confusion_matrix_params or {}

    @property
    def nodes(self) -> list[NodeConfig]:
        """Defines the model nodes, including backbone, neck, and
        head."""
        nodes = [
            NodeConfig(
                name=self.backbone,
                freezing=self._get_freezing(self.backbone_params),
                params=self.backbone_params,
            )
        ]
        if self.use_neck:
            nodes.append(
                NodeConfig(
                    name="RepPANNeck",
                    freezing=self._get_freezing(self.neck_params),
                    inputs=[self.backbone],
                    params=self.neck_params,
                )
            )

        nodes.append(
            NodeConfig(
                name="EfficientKeypointBBoxHead",
                inputs=["RepPANNeck" if self.use_neck else self.backbone],
                freezing=self._get_freezing(self.head_params),
                params=self.head_params,
                task_name=self.task_name,
            )
        )
        return nodes

    @property
    def losses(self) -> list[LossModuleConfig]:
        """Defines the loss module for the keypoint detection task."""
        return [
            LossModuleConfig(
                name="EfficientKeypointBBoxLoss",
                attached_to="EfficientKeypointBBoxHead",
                params=self.loss_params,
                weight=1.0,
            )
        ]

    @property
    def metrics(self) -> list[MetricModuleConfig]:
        """Defines the metrics used for evaluation."""
        metrics = [
            MetricModuleConfig(
                name="ObjectKeypointSimilarity",
                attached_to="EfficientKeypointBBoxHead",
                is_main_metric=True,
            ),
            MetricModuleConfig(
                name="MeanAveragePrecision",
                attached_to="EfficientKeypointBBoxHead",
            ),
        ]
        if self.enable_confusion_matrix:
            metrics.append(
                MetricModuleConfig(
                    name="ConfusionMatrix",
                    attached_to="EfficientKeypointBBoxHead",
                    params={**self.confusion_matrix_params},
                )
            )
        return metrics

    @property
    def visualizers(self) -> list[AttachedModuleConfig]:
        """Defines the visualizer used for the keypoint detection
        task."""
        return [
            AttachedModuleConfig(
                name="KeypointVisualizer",
                attached_to="EfficientKeypointBBoxHead",
                params=self.visualizer_params,
            )
        ]
