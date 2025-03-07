"""MobileNetV3 backbone.

Source: U{<https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.10>}
@license: U{Apple<https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.10/LICENSE>}
"""

from typing import Literal

from loguru import logger
from torch import Tensor, nn

from luxonis_train.nodes.backbones.pplcnet_v3.blocks import make_divisible
from luxonis_train.nodes.base_node import BaseNode
from luxonis_train.nodes.blocks import ConvModule

from .blocks import MobileNetV3ResidualBlock
from .variants import get_variant


class MobileNetV3(BaseNode[Tensor, list[Tensor]]):
    attach_index: int = -1

    def __init__(
        self,
        variant: Literal[
            "SMALL_X0_35",
            "SMALL_X0_5",
            "SMALL_X0_75",
            "SMALL_X1_0",
            "SMALL_X1_25",
            "LARGE_X0_35",
            "LARGE_X0_5",
            "LARGE_X0_75",
            "LARGE_X1_0",
            "LARGE_X1_25",
            "SMALL_X1_0_OS8",
            "LARGE_X1_0_OS8",
        ] = "LARGE_X1_0",
        in_channels: int = 3,
        **kwargs,
    ):
        """MobileNetV3 backbone.

        @type variant: Literal[ "SMALL_X0_35", "SMALL_X0_5",
            "SMALL_X0_75", "SMALL_X1_0", "SMALL_X1_25", "LARGE_X0_35",
            "LARGE_X0_5", "LARGE_X0_75", "LARGE_X1_0", "LARGE_X1_25",
            "SMALL_X1_0_OS8", "LARGE_X1_0_OS8", ]
        @param variant: Specifies which variant of the MobileNetV3
            network to use. Defaults to "LARGE_X1_0".
        @type in_channels: int
        @param in_channels: Number of input channels. Defaults to 3.
        """
        super().__init__(**kwargs)

        var = get_variant(variant)
        self.out_index = var.out_index
        inplanes = 16

        self.conv = ConvModule(
            in_channels=in_channels,
            out_channels=make_divisible(inplanes * var.scale, 8),
            kernel_size=3,
            stride=2,
            padding=1,
            groups=1,
            activation=nn.Hardswish(),
            norm_momentum=0.9,
        )

        self.blocks = nn.Sequential(
            *[
                MobileNetV3ResidualBlock(
                    in_channels=make_divisible(
                        inplanes * var.scale
                        if i == 0
                        else var.net_config[i - 1][2] * var.scale,
                        8,
                    ),
                    mid_channels=make_divisible(var.scale * exp, 8),
                    out_channels=make_divisible(var.scale * c, 8),
                    kernel_size=k,
                    stride=s,
                    use_se=se,
                    activation=act,
                    dilation=td[0] if td else 1,
                )
                for i, (k, exp, c, se, act, s, *td) in enumerate(
                    var.net_config
                )
            ]
        )

        out_channels = [var.net_config[idx][2] for idx in self.out_index]
        self.feat_channels = [
            make_divisible(var.scale * c, 8) for c in out_channels
        ]

        self.init_res(var.stages_pattern)

    def init_res(
        self, stages_pattern, return_patterns=None, return_stages=None
    ):
        if return_patterns and return_stages:
            logger.warning(
                "The 'return_patterns' would be ignored when 'return_stages' is set."
            )
            return_stages = None

        if return_stages is True:
            return_patterns = stages_pattern

        if type(return_stages) is int:
            return_stages = [return_stages]
        if isinstance(return_stages, list):
            if (
                max(return_stages) > len(stages_pattern)
                or min(return_stages) < 0
            ):
                logger.warning(
                    f"The 'return_stages' set error. Illegal value(s) have been ignored. The stages' pattern list is {stages_pattern}."
                )
                return_stages = [
                    val
                    for val in return_stages
                    if val >= 0 and val < len(stages_pattern)
                ]
            return_patterns = [stages_pattern[i] for i in return_stages]

    def forward(self, x):
        x = self.conv(x)

        feat_list = []
        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.out_index:
                feat_list.append(x)

        return feat_list
