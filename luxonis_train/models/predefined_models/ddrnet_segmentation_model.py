from dataclasses import dataclass, field

from luxonis_train.utils.config import (
    LossModuleConfig,
    ModelNodeConfig,
)
from luxonis_train.utils.types import Kwargs

from .segmentation_model import SegmentationModel


@dataclass
class DDRNetSegmentationModel(SegmentationModel):
    backbone: str = "DDRNet"
    highres_planes: int = 64
    layer5_bottleneck_expansion: int = 2
    aux_head_params: Kwargs = field(default_factory=dict)

    @property
    def nodes(self) -> list[ModelNodeConfig]:
        self.backbone_params.update({"highres_planes": self.highres_planes})
        self.backbone_params.update(
            {"layer5_bottleneck_expansion": self.layer5_bottleneck_expansion}
        )

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
        if self.backbone_params.get("use_aux_heads", False):
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
