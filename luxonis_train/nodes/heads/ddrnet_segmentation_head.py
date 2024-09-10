"""DDRNet segmentation head.

Adapted from: U{https://github.com/Deci-AI/super-gradients/blob/master/src/super_gradients/training/models/segmentation_models/ddrnet.py}
Original source: U{https://github.com/ydhongHIT/DDRNet}
Paper: U{https://arxiv.org/pdf/2101.06085.pdf}
@license: U{https://github.com/Deci-AI/super-gradients/blob/master/LICENSE.md}
"""

import torch.nn as nn
from torch import Tensor

from luxonis_train.nodes.base_node import BaseNode
from luxonis_train.nodes.blocks import ConvModule
from luxonis_train.utils.types import LabelType


class DDRNetSegmentationHead(BaseNode[Tensor, Tensor]):
    in_height: int
    in_channels: int
    tasks: list[LabelType] = [LabelType.SEGMENTATION]

    def __init__(
        self,
        num_classes: int,
        in_planes: int = 128,
        inter_planes: int = 64,
        scale_factor: int = 8,
        inter_mode: str = "bilinear",
        attach_index: int = 0,
        **kwargs,
    ):
        """Last stage of the segmentation network.

        @type num_classes: int
        @param num_classes: Output width.
        @type in_planes: int
        @param in_planes: Width of input. Defaults to 128.
        @type inter_planes: int
        @param inter_planes: Width of internal conv. Must be a multiple of
            scale_factor^2 when inter_mode is pixel_shuffle. Defaults to 64.
        @type scale_factor: int
        @param scale_factor: Scaling factor. Defaults to 8.
        @type inter_mode: str
        @param inter_mode: Upsampling method. One of nearest, linear, bilinear, bicubic,
            trilinear, area or pixel_shuffle. If pixel_shuffle is set, nn.PixelShuffle
            is used for scaling. Defaults to "bilinear".
        @type attach_index: int
        @param attach_index: Index at which to attach. Defaults to 0.
        """
        self.attach_index = attach_index
        super().__init__(**kwargs)
        self.scale_factor = scale_factor

        if inter_mode == "pixel_shuffle":
            assert (
                inter_planes % (scale_factor**2) == 0
            ), "When using pixel_shuffle, inter_planes must be a multiple of scale_factor^2."

        self.conv1 = ConvModule(
            in_planes,
            inter_planes,
            kernel_size=3,
            padding=1,
            bias=False,
            activation=nn.ReLU(inplace=True),
        )

        if inter_mode == "pixel_shuffle":
            self.conv2 = ConvModule(
                inter_planes,
                inter_planes,
                kernel_size=1,
                padding=0,
                bias=True,
                activation=nn.Identity(),
            )
            self.upscale = nn.PixelShuffle(scale_factor)
        else:
            self.conv2 = ConvModule(
                inter_planes,
                num_classes,
                kernel_size=1,
                padding=0,
                bias=True,
                activation=nn.Identity(),
            )
            self.upscale = nn.Upsample(scale_factor=scale_factor, mode=inter_mode)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        out = self.conv2(x)
        out = self.upscale(out)

        return out
