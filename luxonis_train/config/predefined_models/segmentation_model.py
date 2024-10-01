from dataclasses import dataclass, field
from typing import Literal

from luxonis_train.config import (
    AttachedModuleConfig,
    LossModuleConfig,
    MetricModuleConfig,
    ModelNodeConfig,
    Params,
)

from .base_predefined_model import BasePredefinedModel


@dataclass
class SegmentationModel(BasePredefinedModel):
    variant: str | None = None
    task: Literal["binary", "multiclass"] = "binary"
    backbone_params: Params = field(default_factory=dict)
    head_params: Params = field(default_factory=dict)
    aux_head_params: Params = field(default_factory=dict)
    loss_params: Params = field(default_factory=dict)
    visualizer_params: Params = field(default_factory=dict)
    task_name: str | None = None

    def __post_init__(self):
        if self.variant == "heavy":
            self.backbone_params.setdefault("variant", "23")
        elif self.variant == "light":
            self.backbone_params.setdefault("variant", "23-slim")

    @property
    def nodes(self) -> list[ModelNodeConfig]:
        self.head_params.update({"attach_index": -1})
        self.aux_head_params.update({"attach_index": -2})
        self.aux_head_params.update(
            {"remove_on_export": True}
        ) if "remove_on_export" not in self.aux_head_params else None

        node_list = [
            ModelNodeConfig(
                name="DDRNet",
                alias="ddrnet_backbone",
                freezing=self.backbone_params.pop("freezing", {}),
                params=self.backbone_params,
            ),
            ModelNodeConfig(
                name="DDRNetSegmentationHead",
                alias="segmentation_head",
                inputs=["ddrnet_backbone"],
                freezing=self.head_params.pop("freezing", {}),
                params=self.head_params,
                task=self.task_name,
            ),
        ]
        if self.backbone_params.get("use_aux_heads", True):
            node_list.append(
                ModelNodeConfig(
                    name="DDRNetSegmentationHead",
                    alias="aux_segmentation_head",
                    inputs=["ddrnet_backbone"],
                    freezing=self.aux_head_params.pop("freezing", {}),
                    params=self.aux_head_params,
                    task=self.task_name,
                )
            )
        return node_list

    @property
    def losses(self) -> list[LossModuleConfig]:
        loss_list = [
            LossModuleConfig(
                name="BCEWithLogitsLoss"
                if self.task == "binary"
                else "CrossEntropyLoss",
                alias="segmentation_loss",
                attached_to="segmentation_head",
                params=self.loss_params,
                weight=1.0,
            ),
        ]
        if self.backbone_params.get("use_aux_heads", False):
            loss_list.append(
                LossModuleConfig(
                    name="BCEWithLogitsLoss"
                    if self.task == "binary"
                    else "CrossEntropyLoss",
                    alias="aux_segmentation_loss",
                    attached_to="aux_segmentation_head",
                    params=self.loss_params,
                    weight=0.4,
                )
            )
        return loss_list

    @property
    def metrics(self) -> list[MetricModuleConfig]:
        return [
            MetricModuleConfig(
                name="JaccardIndex",
                alias="segmentation_jaccard_index",
                attached_to="segmentation_head",
                is_main_metric=True,
                params={"task": self.task},
            ),
            MetricModuleConfig(
                name="F1Score",
                alias="segmentation_f1_score",
                attached_to="segmentation_head",
                params={"task": self.task},
            ),
        ]

    @property
    def visualizers(self) -> list[AttachedModuleConfig]:
        return [
            AttachedModuleConfig(
                name="SegmentationVisualizer",
                alias="segmentation_visualizer",
                attached_to="segmentation_head",
                params=self.visualizer_params,
            )
        ]
