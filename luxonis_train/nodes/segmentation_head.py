"""Implementation of a basic segmentation head.

Adapted from: U{https://github.com/pytorch/vision/blob/main/torchvision/models/segmentation/fcn.py}
@license: U{BSD-3 <https://github.com/pytorch/vision/blob/main/LICENSE>}
"""


import torch.nn as nn
from torch import Tensor

from luxonis_train.nodes.blocks import UpBlock
from luxonis_train.utils.general import infer_upscale_factor
from luxonis_train.utils.types import LabelType, Packet

from .base_node import BaseNode


class SegmentationHead(BaseNode[Tensor, Tensor]):
    attach_index: int = -1
    in_height: int
    in_channels: int

    def __init__(self, **kwargs):
        """Basic segmentation FCN head.

        Note that it doesn't ensure that ouptut is same size as input.

        @type kwargs: Any
        @param kwargs: Additional arguments to pass to L{BaseNode}.
        """
        super().__init__(task_type=LabelType.SEGMENTATION, **kwargs)

        original_height = self.original_in_shape[2]
        num_up = infer_upscale_factor(self.in_height, original_height, strict=False)

        modules = []
        in_channels = self.in_channels
        for _ in range(int(num_up)):
            modules.append(
                UpBlock(in_channels=in_channels, out_channels=in_channels // 2)
            )
            in_channels //= 2

        self.head = nn.Sequential(
            *modules,
            nn.Conv2d(in_channels, self.n_classes, kernel_size=1),
        )

    def wrap(self, output: Tensor) -> Packet[Tensor]:
        return {"segmentation": [output]}

    def forward(self, inputs: Tensor) -> Tensor:
        return self.head(inputs)
