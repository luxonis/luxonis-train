from typing import Literal, TypeAlias

from pydantic import BaseModel

from luxonis_train.config import (
    AttachedModuleConfig,
    LossModuleConfig,
    ModelNodeConfig,
    Params,
)

from .base_predefined_model import BasePredefinedModel

VariantLiteral: TypeAlias = Literal["light", "heavy"]


class AnomalyVariant(BaseModel):
    variant: str


def get_variant(variant: VariantLiteral) -> AnomalyVariant:
    """Returns the specific variant configuration for the
    AnomalyDetectionModel."""
    variants = {
        "light": AnomalyVariant(
            variant="N",
        ),
        "heavy": AnomalyVariant(
            variant="L",
        ),
    }

    if variant not in variants:
        raise ValueError(
            f"Anomaly detection variant should be one of {list(variants.keys())}, got '{variant}'."
        )

    return variants[variant]


class AnomalyDetectionModel(BasePredefinedModel):
    def __init__(
        self,
        variant: VariantLiteral = "ligth",
        rec_subnet_params: Params | None = None,
        disc_subnet_params: Params | None = None,
        loss_params: Params | None = None,
        visualizer_params: Params | None = None,
        task_name: str | None = None,
    ):
        var_config = get_variant(variant)

        self.rec_subnet_params = (
            rec_subnet_params
            if rec_subnet_params is not None
            else {"variant": var_config.variant}
        )
        self.disc_subnet_params = (
            disc_subnet_params
            if disc_subnet_params is not None
            else {"variant": var_config.variant}
        )
        self.loss_params = loss_params or {}
        self.visualizer_params = visualizer_params or {}
        self.task_name = task_name or "segmentation"

    @property
    def nodes(self) -> list[ModelNodeConfig]:
        """Defines the model nodes, including RecSubNet and
        DiscSubNetHead."""
        nodes = [
            ModelNodeConfig(
                name="RecSubNet",
                alias="rec_subnet",
                params=self.rec_subnet_params,
            ),
            ModelNodeConfig(
                name="DiscSubNetHead",
                alias="disc_subnet_head",
                inputs=["rec_subnet"],
                params=self.disc_subnet_params,
            ),
        ]
        return nodes

    @property
    def losses(self) -> list[LossModuleConfig]:
        """Defines the loss module for the anomaly detection task."""
        return [
            LossModuleConfig(
                name="ReconstructionSegmentationLoss",
                alias="anomaly_loss",
                attached_to="disc_subnet_head",
                params=self.loss_params,
                weight=1.0,
            )
        ]

    @property
    def visualizers(self) -> list[AttachedModuleConfig]:
        """Defines the visualizer used for the anomaly detection
        task."""
        return [
            AttachedModuleConfig(
                name="SegmentationVisualizer",
                alias="anomaly_visualizer",
                attached_to="disc_subnet_head",
                params=self.visualizer_params,
            )
        ]
