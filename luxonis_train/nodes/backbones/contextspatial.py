import torch.nn.functional as F
from luxonis_ml.typing import Kwargs
from torch import Tensor, nn

from luxonis_train.nodes.base_node import BaseNode
from luxonis_train.nodes.blocks import (
    AttentionRefinmentBlock,
    ConvBlock,
    FeatureFusionBlock,
)
from luxonis_train.registry import NODES


class ContextSpatial(BaseNode):
    def __init__(
        self,
        context_backbone: str | nn.Module = "MobileNetV2",
        backbone_kwargs: Kwargs | None = None,
        **kwargs,
    ):
        """Context Spatial backbone introduced in BiseNetV1.

        Source: U{BiseNetV1<https://github.com/taveraantonio/BiseNetv1>}

        @see: U{BiseNetv1: Bilateral Segmentation Network for
            Real-time Semantic Segmentation
            <https://arxiv.org/abs/1808.00897>}

        @type context_backbone: str
        @param context_backbone: Backbone used in the context path.
            Can be either a string or a C{nn.Module}.
            If a string argument is used, it has to be a name of a module
            stored in the L{NODES} registry. Defaults to C{MobileNetV2}.

        @type backbone_kwargs: dict
        @param backbone_kwargs: Keyword arguments for the backbone.
            Only used when the C{context_backbone} argument is a string.
        """
        super().__init__(**kwargs)

        if isinstance(context_backbone, str):
            backbone_kwargs = backbone_kwargs or {}
            backbone_kwargs |= kwargs
            context_backbone = NODES.get(context_backbone)(**backbone_kwargs)

        self.context_path = ContextPath(context_backbone)
        self.spatial_path = SpatialPath(3, 128)
        self.ffm = FeatureFusionBlock(256, 256)

    def forward(self, inputs: Tensor) -> list[Tensor]:
        spatial_out = self.spatial_path(inputs)
        context16, _ = self.context_path(inputs)
        fm_fuse = self.ffm(spatial_out, context16)
        return [fm_fuse]


class SpatialPath(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        intermediate_channels = 64
        self.conv_7x7 = ConvBlock(
            in_channels,
            intermediate_channels,
            kernel_size=7,
            stride=2,
            padding=3,
        )
        self.conv_3x3_1 = ConvBlock(
            intermediate_channels,
            intermediate_channels,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.conv_3x3_2 = ConvBlock(
            intermediate_channels,
            intermediate_channels,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.conv_1x1 = ConvBlock(
            intermediate_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_7x7(x)
        x = self.conv_3x3_1(x)
        x = self.conv_3x3_2(x)
        return self.conv_1x1(x)


class ContextPath(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

        self.up16 = nn.Upsample(
            scale_factor=2.0, mode="bilinear", align_corners=True
        )
        self.up32 = nn.Upsample(
            scale_factor=2.0, mode="bilinear", align_corners=True
        )

        self.refine16 = ConvBlock(128, 128, 3, 1, 1)
        self.refine32 = ConvBlock(128, 128, 3, 1, 1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        *_, down16, down32 = self.backbone(x)

        if not hasattr(self, "arm16"):
            self.arm16 = AttentionRefinmentBlock(down16.shape[1], 128)
            self.arm32 = AttentionRefinmentBlock(down32.shape[1], 128)

            self.global_context = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                ConvBlock(down32.shape[1], 128, 1, 1, 0),
            )

        arm_down16 = self.arm16(down16)
        arm_down32 = self.arm32(down32)

        global_down32 = self.global_context(down32)
        global_down32 = F.interpolate(
            global_down32,
            size=down32.shape[2:],
            mode="bilinear",
            align_corners=True,
        )

        arm_down32 += global_down32
        arm_down32 = self.up32(arm_down32)
        arm_down32 = self.refine32(arm_down32)

        arm_down16 += arm_down32
        arm_down16 = self.up16(arm_down16)
        arm_down16 = self.refine16(arm_down16)

        return arm_down16, arm_down32
