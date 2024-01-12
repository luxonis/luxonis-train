"""Implementation of Context Spatial backbone.

Source: U{BiseNetV1<https://github.com/taveraantonio/BiseNetv1>}
"""


from torch import Tensor, nn
from torch.nn import functional as F

from luxonis_train.nodes.blocks import (
    AttentionRefinmentBlock,
    ConvModule,
    FeatureFusionBlock,
)
from luxonis_train.utils.registry import NODES

from .base_node import BaseNode


class ContextSpatial(BaseNode[Tensor, list[Tensor]]):
    attach_index: int = -1

    def __init__(self, context_backbone: str = "MobileNetV2", **kwargs):
        """Context spatial backbone.
        TODO: Add more documentation.


        @type context_backbone: str
        @param context_backbone: Backbone used. Defaults to C{MobileNetV2}.
        """
        super().__init__(**kwargs)

        self.context_path = ContextPath(NODES.get(context_backbone)(**kwargs))
        self.spatial_path = SpatialPath(3, 128)
        self.ffm = FeatureFusionBlock(256, 256)

    def forward(self, x: Tensor) -> list[Tensor]:
        spatial_out = self.spatial_path(x)
        context16, _ = self.context_path(x)
        fm_fuse = self.ffm(spatial_out, context16)
        outs = [fm_fuse]
        return outs


class SpatialPath(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        intermediate_channels = 64
        self.conv_7x7 = ConvModule(in_channels, intermediate_channels, 7, 2, 3)
        self.conv_3x3_1 = ConvModule(
            intermediate_channels, intermediate_channels, 3, 2, 1
        )
        self.conv_3x3_2 = ConvModule(
            intermediate_channels, intermediate_channels, 3, 2, 1
        )
        self.conv_1x1 = ConvModule(intermediate_channels, out_channels, 1, 1, 0)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_7x7(x)
        x = self.conv_3x3_1(x)
        x = self.conv_3x3_2(x)
        return self.conv_1x1(x)


class ContextPath(nn.Module):
    def __init__(self, backbone: BaseNode):
        super().__init__()
        self.backbone = backbone

        self.up16 = nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=True)
        self.up32 = nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=True)

        self.refine16 = ConvModule(128, 128, 3, 1, 1)
        self.refine32 = ConvModule(128, 128, 3, 1, 1)

    def forward(self, x: Tensor) -> list[Tensor]:
        *_, down16, down32 = self.backbone.forward(x)

        if not hasattr(self, "arm16"):
            self.arm16 = AttentionRefinmentBlock(down16.shape[1], 128)
            self.arm32 = AttentionRefinmentBlock(down32.shape[1], 128)

            self.global_context = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), ConvModule(down32.shape[1], 128, 1, 1, 0)
            )

        arm_down16 = self.arm16(down16)
        arm_down32 = self.arm32(down32)

        global_down32 = self.global_context(down32)
        global_down32 = F.interpolate(
            global_down32, size=down32.size()[2:], mode="bilinear", align_corners=True
        )

        arm_down32 = arm_down32 + global_down32
        arm_down32 = self.up32(arm_down32)
        arm_down32 = self.refine32(arm_down32)

        arm_down16 = arm_down16 + arm_down32
        arm_down16 = self.up16(arm_down16)
        arm_down16 = self.refine16(arm_down16)

        return [arm_down16, arm_down32]
