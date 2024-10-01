from dataclasses import dataclass, field

from luxonis_train.config import (
    AttachedModuleConfig,
    LossModuleConfig,
    MetricModuleConfig,
    ModelNodeConfig,
    Params,
)

from .base_predefined_model import BasePredefinedModel


@dataclass
class DetectionModel(BasePredefinedModel):
    use_neck: bool = True
    variant: str | None = None
    backbone_params: Params = field(default_factory=dict)
    neck_params: Params = field(default_factory=dict)
    head_params: Params = field(default_factory=dict)
    loss_params: Params = field(default_factory=dict)
    visualizer_params: Params = field(default_factory=dict)
    task_name: str | None = None

    def __post_init__(self):
        if self.variant == "heavy":
            self.backbone_params.setdefault("variant", "l")
        elif self.variant == "light":
            self.backbone_params.setdefault("variant", "n")
        self.loss_params.setdefault("n_warmup_epochs", 0)

    @property
    def nodes(self) -> list[ModelNodeConfig]:
        nodes = [
            ModelNodeConfig(
                name="EfficientRep",
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
        return [
            AttachedModuleConfig(
                name="BBoxVisualizer",
                alias="detection_visualizer",
                attached_to="detection_head",
                params=self.visualizer_params,
            )
        ]
