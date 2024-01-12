"""BiSeNet segmentation head.

Adapted from U{https://github.com/taveraantonio/BiseNetv1}.
License: NOT SPECIFIED.
"""


from torch import Tensor, nn

from luxonis_train.nodes.blocks import ConvModule
from luxonis_train.utils.general import infer_upscale_factor
from luxonis_train.utils.types import LabelType, Packet

from .base_node import BaseNode


class BiSeNetHead(BaseNode[Tensor, Tensor]):
    attach_index: int = -1
    in_height: int
    in_channels: int

    def __init__(
        self,
        intermediate_channels: int = 64,
        **kwargs,
    ):
        """BiSeNet segmentation head.
        TODO: Add more documentation.

        @type intermediate_channels: int
        @param intermediate_channels: How many intermediate channels to use.
            Defaults to C{64}.
        """
        super().__init__(task_type=LabelType.SEGMENTATION, **kwargs)

        original_height = self.original_in_shape[2]
        upscale_factor = 2 ** infer_upscale_factor(self.in_height, original_height)
        out_channels = self.n_classes * upscale_factor * upscale_factor

        self.conv_3x3 = ConvModule(self.in_channels, intermediate_channels, 3, 1, 1)
        self.conv_1x1 = nn.Conv2d(intermediate_channels, out_channels, 1, 1, 0)
        self.upscale = nn.PixelShuffle(upscale_factor)

    def wrap(self, output: Tensor) -> Packet[Tensor]:
        return {"segmentation": [output]}

    def forward(self, inputs: Tensor) -> Tensor:
        inputs = self.conv_3x3(inputs)
        inputs = self.conv_1x1(inputs)
        return self.upscale(inputs)
