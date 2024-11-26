from typing import Any

from torch import Tensor, nn

from luxonis_train.enums import TaskType
from luxonis_train.nodes.base_node import BaseNode
from luxonis_train.nodes.blocks import ConvModule
from luxonis_train.utils import infer_upscale_factor


class BiSeNetHead(BaseNode[Tensor, Tensor]):
    in_height: int
    in_width: int
    in_channels: int

    tasks: list[TaskType] = [TaskType.SEGMENTATION]

    def __init__(self, intermediate_channels: int = 64, **kwargs: Any):
        """BiSeNet segmentation head.

        Source: U{BiseNetV1<https://github.com/taveraantonio/BiseNetv1>}
        @license: NOT SPECIFIED.
        @see: U{BiseNetv1: Bilateral Segmentation Network for
            Real-time Semantic Segmentation
            <https://arxiv.org/abs/1808.00897>}

        @type intermediate_channels: int
        @param intermediate_channels: How many intermediate channels to use.
            Defaults to C{64}.
        """
        super().__init__(**kwargs)

        h, w = self.original_in_shape[1:]
        upscale_factor = 2 ** infer_upscale_factor(
            (self.in_height, self.in_width), (h, w)
        )
        out_channels = self.n_classes * upscale_factor * upscale_factor

        self.conv_3x3 = ConvModule(
            self.in_channels,
            intermediate_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv_1x1 = nn.Conv2d(
            intermediate_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.upscale = nn.PixelShuffle(upscale_factor)

    def forward(self, inputs: Tensor) -> Tensor:
        x = self.conv_3x3(inputs)
        x = self.conv_1x1(x)
        return self.upscale(x)
