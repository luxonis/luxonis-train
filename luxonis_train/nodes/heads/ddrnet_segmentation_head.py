"""DDRNet segmentation head.

Adapted from: U{https://github.com/Deci-AI/super-gradients/blob/master/src/super_gradients/training/models/segmentation_models/ddrnet.py}
Original source: U{https://github.com/ydhongHIT/DDRNet}
Paper: U{https://arxiv.org/pdf/2101.06085.pdf}
@license: U{https://github.com/Deci-AI/super-gradients/blob/master/LICENSE.md}
"""

import torch.nn as nn
from torch import Tensor

from luxonis_train.nodes.base_node import BaseNode
from luxonis_train.nodes.blocks import UpBlock
from luxonis_train.utils.general import infer_upscale_factor
from luxonis_train.utils.types import LabelType


class DDRNetSegmentationHead(BaseNode[Tensor, Tensor]):
    in_height: int
    in_channels: int
    tasks: list[LabelType] = [LabelType.SEGMENTATION]

    def __init__(self, num_classes: int, in_planes: int = 128, inter_planes: int = 64, scale_factor: int = 8, inter_mode: str = "bilinear", attach_index=0, **kwargs):
        """
        Last stage of the segmentation network.
        Reduces the number of output planes (usually to num_classes) while increasing the size by scale_factor
        :param in_planes: width of input
        :param inter_planes: width of internal conv. must be a multiple of scale_factor^2 when inter_mode=pixel_shuffle
        :param num_classes: output width
        :param scale_factor: scaling factor
        :param inter_mode: one of nearest, linear, bilinear, bicubic, trilinear, area or pixel_shuffle.
        when set to pixel_shuffle, an nn.PixelShuffle will be used for scaling
        """
        self.attach_index = attach_index

        super().__init__(**kwargs)

        if inter_mode == "pixel_shuffle":
            assert inter_planes % (scale_factor ^ 2) == 0, "when using pixel_shuffle, inter_planes must be a multiple of scale_factor^2"

        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.relu = nn.ReLU(inplace=True)

        if inter_mode == "pixel_shuffle":
            self.conv2 = nn.Conv2d(inter_planes, inter_planes, kernel_size=1, padding=0, bias=True)
            self.upscale = nn.PixelShuffle(scale_factor)
        else:
            self.conv2 = nn.Conv2d(inter_planes, num_classes, kernel_size=1, padding=0, bias=True)
            self.upscale = nn.Upsample(scale_factor=scale_factor, mode=inter_mode)

        self.scale_factor = scale_factor

    def forward(self, x):
        x = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(x)))
        out = self.upscale(out)

        return out
