from typing import Literal

from luxonis_ml.typing import Params
from pydantic import BaseModel

from luxonis_train.config import (
    AttachedModuleConfig,
    LossModuleConfig,
    MetricModuleConfig,
    NodeConfig,
)

from .base_predefined_model import BasePredefinedModel

VariantLiteral = Literal["light", "heavy"]


class SegmentationVariant(BaseModel):
    backbone: str
    backbone_params: Params
    head_params: Params


def get_variant(variant: VariantLiteral) -> SegmentationVariant:
    """Returns the specific variant configuration for the
    SegmentationModel."""
    variants = {
        "light": SegmentationVariant(
            backbone="DDRNet",
            backbone_params={"variant": "23-slim"},
            head_params={"download_weights": True},
        ),
        "heavy": SegmentationVariant(
            backbone="DDRNet",
            backbone_params={"variant": "23"},
            head_params={"download_weights": True},
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
        task: Literal["binary", "multiclass"] | None = None,
        task_name: str = "",
        enable_confusion_matrix: bool = True,
        confusion_matrix_params: Params | None = None,
    ):
        var_config = get_variant(variant)

        self.backbone = backbone or var_config.backbone
        self.backbone_params = (
            backbone_params
            if backbone is not None or backbone_params is not None
            else var_config.backbone_params
        ) or {}
        self.head_params = head_params or var_config.head_params
        self.aux_head_params = aux_head_params or {}
        self.loss_params = loss_params or {}
        self.visualizer_params = visualizer_params or {}
        self.task = task
        self.task_name = task_name
        self.enable_confusion_matrix = enable_confusion_matrix
        self.confusion_matrix_params = confusion_matrix_params or {}

    @property
    def nodes(self) -> list[NodeConfig]:
        """Defines the model nodes, including backbone and head."""
        self.head_params.update({"attach_index": -1})
        self.aux_head_params.update({"attach_index": -2})

        node_list = [
            NodeConfig(
                name=self.backbone,
                freezing=self._get_freezing(self.backbone_params),
                params=self.backbone_params,
                task_name=self.task_name,
            ),
            NodeConfig(
                name="DDRNetSegmentationHead",
                freezing=self._get_freezing(self.head_params),
                inputs=[self.backbone],
                params=self.head_params,
                task_name=self.task_name,
            ),
        ]
        if self.backbone_params.get("use_aux_heads", True):
            remove_on_export = self.aux_head_params.pop(
                "remove_on_export", True
            )
            if not isinstance(remove_on_export, bool):
                raise ValueError(
                    "The 'remove_on_export' parameter must be a boolean. "
                    f"Got `{remove_on_export}`."
                )
            node_list.append(
                NodeConfig(
                    name="DDRNetSegmentationHead",
                    freezing=self._get_freezing(self.aux_head_params),
                    inputs=[self.backbone],
                    params=self.aux_head_params,
                    task_name=self.task_name,
                    remove_on_export=remove_on_export,
                )
            )
        return node_list

    @property
    def losses(self) -> list[LossModuleConfig]:
        """Defines the loss module for the segmentation task."""
        loss_list = [
            LossModuleConfig(
                name=(
                    "OHEMBCEWithLogitsLoss"
                    if self.task == "binary"
                    else "OHEMCrossEntropyLoss"
                ),
                attached_to="DDRNetSegmentationHead",
                params=self.loss_params,
                weight=1.0,
            )
        ]
        if self.backbone_params.get("use_aux_heads", False):
            loss_list.append(
                LossModuleConfig(
                    name=(
                        "OHEMBCEWithLogitsLoss"
                        if self.task == "binary"
                        else "OHEMCrossEntropyLoss"
                    ),
                    attached_to="DDRNetSegmentationHead_aux",
                    params=self.loss_params,
                    weight=0.4,
                )
            )
        return loss_list

    @property
    def metrics(self) -> list[MetricModuleConfig]:
        """Defines the metrics used for evaluation."""
        metrics = [
            MetricModuleConfig(
                name="JaccardIndex",
                attached_to="DDRNetSegmentationHead",
                is_main_metric=True,
                params={"task": self.task},
            ),
            MetricModuleConfig(
                name="F1Score",
                attached_to="DDRNetSegmentationHead",
                params={"task": self.task},
            ),
        ]
        if self.enable_confusion_matrix:
            metrics.append(
                MetricModuleConfig(
                    name="ConfusionMatrix",
                    attached_to="DDRNetSegmentationHead",
                    params={**self.confusion_matrix_params},
                )
            )
        return metrics

    @property
    def visualizers(self) -> list[AttachedModuleConfig]:
        """Defines the visualizer used for the segmentation task."""
        return [
            AttachedModuleConfig(
                name="SegmentationVisualizer",
                attached_to="DDRNetSegmentationHead",
                params=self.visualizer_params,
            )
        ]
