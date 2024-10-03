from typing import Literal

from pydantic import BaseModel

from luxonis_train.config import (
    AttachedModuleConfig,
    LossModuleConfig,
    MetricModuleConfig,
    ModelNodeConfig,
    Params,
)

from .base_predefined_model import BasePredefinedModel

VariantLiteral = Literal["light", "heavy"]


class SegmentationVariant(BaseModel):
    backbone: str
    backbone_params: Params


def get_variant(variant: VariantLiteral) -> SegmentationVariant:
    """Returns the specific variant configuration for the
    SegmentationModel."""
    variants = {
        "light": SegmentationVariant(
            backbone="DDRNet",
            backbone_params={"variant": "23-slim"},
        ),
        "heavy": SegmentationVariant(
            backbone="DDRNet",
            backbone_params={"variant": "23"},
        ),
    }

    if variant not in variants:
        raise ValueError(
            f"Segmentation variant should be one of {list(variants.keys())}, got '{variant}'."
        )

    return variants[variant]


class SegmentationModel(BasePredefinedModel):
    def __init__(
        self,
        variant: VariantLiteral = "light",
        backbone: str | None = None,
        backbone_params: Params | None = None,
        head_params: Params | None = None,
        aux_head_params: Params | None = None,
        loss_params: Params | None = None,
        visualizer_params: Params | None = None,
        task: Literal["binary", "multiclass"] = "binary",
        task_name: str | None = None,
    ):
        var_config = get_variant(variant)

        self.backbone = backbone or var_config.backbone
        self.backbone_params = (
            backbone_params
            if backbone is not None or backbone_params is not None
            else var_config.backbone_params
        ) or {}
        self.head_params = head_params or {}
        self.aux_head_params = aux_head_params or {}
        self.loss_params = loss_params or {}
        self.visualizer_params = visualizer_params or {}
        self.task = task
        self.task_name = task_name or "segmentation"

    @property
    def nodes(self) -> list[ModelNodeConfig]:
        """Defines the model nodes, including backbone and head."""
        self.head_params.update({"attach_index": -1})
        self.aux_head_params.update({"attach_index": -2})

        node_list = [
            ModelNodeConfig(
                name=self.backbone,
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
                    remove_on_export=self.aux_head_params.pop(
                        "remove_on_export", True
                    ),
                )
            )
        return node_list

    @property
    def losses(self) -> list[LossModuleConfig]:
        """Defines the loss module for the segmentation task."""
        loss_list = [
            LossModuleConfig(
                name=(
                    "BCEWithLogitsLoss"
                    if self.task == "binary"
                    else "CrossEntropyLoss"
                ),
                alias="segmentation_loss",
                attached_to="segmentation_head",
                params=self.loss_params,
                weight=1.0,
            ),
        ]
        if self.backbone_params.get("use_aux_heads", False):
            loss_list.append(
                LossModuleConfig(
                    name=(
                        "BCEWithLogitsLoss"
                        if self.task == "binary"
                        else "CrossEntropyLoss"
                    ),
                    alias="aux_segmentation_loss",
                    attached_to="aux_segmentation_head",
                    params=self.loss_params,
                    weight=0.4,
                )
            )
        return loss_list

    @property
    def metrics(self) -> list[MetricModuleConfig]:
        """Defines the metrics used for evaluation."""
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
        """Defines the visualizer used for the segmentation task."""
        return [
            AttachedModuleConfig(
                name="SegmentationVisualizer",
                alias="segmentation_visualizer",
                attached_to="segmentation_head",
                params=self.visualizer_params,
            )
        ]
