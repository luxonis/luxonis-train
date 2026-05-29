from luxonis_ml.typing import Params
from torch import Tensor, nn
from typing_extensions import override

from luxonis_train.nodes.blocks import ConvBlock
from luxonis_train.nodes.heads import BaseHead
from luxonis_train.tasks import Tasks
from luxonis_train.utils import infer_upscale_factor


class BiSeNetHead(BaseHead):
    in_height: int
    in_width: int
    in_channels: int

    task = Tasks.SEGMENTATION
    parser: str = "SegmentationParser"

    def __init__(self, intermediate_channels: int = 64, **kwargs):
        """BiSeNet segmentation head.

        Source: `BiseNetV1 <https://github.com/taveraantonio/BiseNetv1>`_

        Args:
            intermediate_channels (int): How many intermediate channels to use. Defaults to ``64``.

        Notes:
            License: NOT SPECIFIED.

        See Also:
            `BiseNetv1: Bilateral Segmentation Network for Real-time Semantic Segmentation <https://arxiv.org/abs/1808.00897>`_
        """
        super().__init__(**kwargs)

        h, w = self.original_in_shape[1:]
        upscale_factor = 2 ** infer_upscale_factor(
            (self.in_height, self.in_width), (h, w)
        )
        out_channels = self.n_classes * upscale_factor * upscale_factor

        self.conv_3x3 = ConvBlock(
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

    @override
    def get_custom_head_config(self) -> Params:
        return {"is_softmax": False}
