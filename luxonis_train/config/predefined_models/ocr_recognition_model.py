from typing import Literal, TypeAlias

from pydantic import BaseModel

from luxonis_train.config import (
    AttachedModuleConfig,
    LossModuleConfig,
    MetricModuleConfig,
    ModelNodeConfig,
    Params,
)
from luxonis_train.utils.ocr import AlphabetName

from .base_predefined_model import BasePredefinedModel

VariantLiteral: TypeAlias = Literal["light", "heavy"]


class OCRRecognitionVariant(BaseModel):
    backbone: str
    backbone_params: Params
    neck_params: Params
    head_params: Params


def get_variant(variant: VariantLiteral) -> OCRRecognitionVariant:
    """Returns the specific variant configuration for the
    OCRRecognitionModel."""
    variants = {
        "light": OCRRecognitionVariant(
            backbone="PPLCNetV3",
            backbone_params={"variant": "rec-light"},
            neck_params={},
            head_params={},
        ),
    }

    if variant not in variants:
        raise ValueError(
            f"OCR variant should be one of {list(variants.keys())}, got '{variant}'."
        )

    return variants[variant]


class OCRRecognitionModel(BasePredefinedModel):
    """A predefined model for OCR recognition tasks."""

    def __init__(
        self,
        variant: VariantLiteral = "light",
        backbone: str | None = None,
        backbone_params: Params | None = None,
        neck_params: Params | None = None,
        head_params: Params | None = None,
        loss_params: Params | None = None,
        visualizer_params: Params | None = None,
        task_name: str | None = None,
        alphabet: list[str] | AlphabetName = "english",
        max_text_len: int = 40,
        ignore_unknown: bool = True,
    ):
        var_config = get_variant(variant)

        self.backbone = backbone or var_config.backbone
        self.backbone_params = (
            backbone_params
            if backbone is not None or backbone_params is not None
            else var_config.backbone_params
        ) or {}
        self.neck_params = neck_params or var_config.neck_params
        self.head_params = head_params or var_config.head_params

        self.backbone_params["max_text_len"] = max_text_len
        self.head_params["alphabet"] = alphabet
        self.head_params["ignore_unknown"] = ignore_unknown
        self.loss_params = loss_params or {}
        self.visualizer_params = visualizer_params or {}
        self.task_name = task_name or "ocr_recognition"

    @property
    def nodes(self) -> list[ModelNodeConfig]:
        """Defines the model nodes, including backbone and head."""
        return [
            ModelNodeConfig(
                name=self.backbone,
                alias=f"{self.task_name}/{self.backbone}",
                freezing=self.backbone_params.pop("freezing", {}),
                params=self.backbone_params,
            ),
            ModelNodeConfig(
                name="SVTRNeck",
                alias=f"{self.task_name}/SVTRNeck",
                inputs=[f"{self.task_name}/{self.backbone}"],
                freezing=self.neck_params.pop("freezing", {}),
                params=self.neck_params,
            ),
            ModelNodeConfig(
                name="OCRCTCHead",
                alias=f"{self.task_name}/OCRCTCHead",
                inputs=[f"{self.task_name}/SVTRNeck"],
                freezing=self.head_params.pop("freezing", {}),
                params=self.head_params,
            ),
        ]

    @property
    def losses(self) -> list[LossModuleConfig]:
        """Defines the loss module for the classification task."""
        return [
            LossModuleConfig(
                name="CTCLoss",
                alias=f"{self.task_name}/CTCLoss",
                attached_to=f"{self.task_name}/OCRCTCHead",
                params=self.loss_params,
                weight=1.0,
            )
        ]

    @property
    def metrics(self) -> list[MetricModuleConfig]:
        """Defines the metrics used for evaluation."""
        metrics = [
            MetricModuleConfig(
                name="OCRAccuracy",
                attached_to=f"{self.task_name}/OCRCTCHead",
                is_main_metric=True,
            ),
        ]
        return metrics

    @property
    def visualizers(self) -> list[AttachedModuleConfig]:
        """Defines the visualizer used for the detection task."""
        return [
            AttachedModuleConfig(
                name="OCRVisualizer",
                attached_to=f"{self.task_name}/OCRCTCHead",
                params=self.visualizer_params,
            )
        ]
