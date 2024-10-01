from dataclasses import dataclass

from luxonis_train.config import LossModuleConfig, ModelNodeConfig

from .segmentation_model import SegmentationModel


@dataclass
class DDRNetSegmentationModel(SegmentationModel):
    variant: str | None = None

    def __post_init__(self):
        if self.variant == "heavy":
            self.backbone_params.setdefault("variant", "23")
        elif self.variant == "light":
            self.backbone_params.setdefault("variant", "23-slim")

    @property
    def nodes(self) -> list[ModelNodeConfig]:
        self.head_params.update({"attach_index": -1})

        self.aux_head_params.update({"attach_index": -2})

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
