from typing import Literal

import torch.nn.functional as F
from torch import Tensor, nn

from luxonis_train.nodes.base_node import BaseNode
from luxonis_train.nodes.blocks import ConvModule

from .blocks import LCNetV3Block
from .variants import get_variant


def make_divisible(
    v: int | float, divisor: int = 16, min_value: int | None = None
) -> int:
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class PPLCNetV3(BaseNode[Tensor, list[Tensor]]):
    in_channels: int

    def __init__(
        self,
        variant: Literal["rec-light"] = "rec-light",
        scale: float | None = None,
        n_branches: int | None = None,
        use_detection_backbone: bool | None = None,
        max_text_len: int = 40,
        **kwargs,
    ):
        """PPLCNetV3 backbone.

        @see: U{Adapted from <https://github.com/PaddlePaddle/PaddleOCR/
            blob/main/ppocr/modeling/backbones/rec_lcnetv3.py>}
        @see: U{Original code
            <https://github.com/PaddlePaddle/PaddleOCR>}
        @license: U{Apache License, Version 2.0
            <https://github.com/PaddlePaddle/PaddleOCR/blob/main/LICENSE
            >}
        @type scale: float
        @param scale: Scale factor. Defaults to 0.95.
        @type n_branches: int
        @param n_branches: Number of convolution branches.
            Defaults to 4.
        @type use_detection_backbone: bool
        @param use_detection_backbone: Whether to use the detection backbone.
            Defaults to False.
        @type max_text_len: int
        @param max_text_len: Maximum text length. Defaults to 40.
        """
        super().__init__(**kwargs)

        var = get_variant(variant)

        self.scale = scale or var.scale
        self.use_detection_backbone = (
            use_detection_backbone or var.use_detection_backbone
        )
        self.n_branches = n_branches or var.n_branches
        self.net_config = var.net_config

        self.max_text_len = max_text_len

        self.conv = ConvModule(
            in_channels=self.in_channels,
            out_channels=make_divisible(16 * self.scale),
            kernel_size=3,
            stride=2,
            padding=1,
            bias=True,
            activation=nn.Identity(),
        )
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    *[
                        LCNetV3Block(
                            in_channels=make_divisible(
                                b.in_channels * self.scale
                            ),
                            out_channels=make_divisible(
                                b.out_channels * self.scale
                            ),
                            kernel_size=b.kernel_size,
                            stride=b.stride,
                            use_se=b.use_se,
                            n_branches=self.n_branches,
                        )
                        for b in net_config
                    ]
                )
                for net_config in self.net_config
            ]
        )

        if self.use_detection_backbone:
            blocks_out_channels = [
                make_divisible(
                    self.net_config[i][-1].out_channels * self.scale
                )
                for i in range(1, 5)
            ]

            detecion_out_channels = [
                int(c * self.scale) for c in [16, 24, 56, 480]
            ]

            self.detecion_blocks = nn.ModuleList(
                [
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=True,
                    )
                    for in_channels, out_channels in zip(
                        blocks_out_channels, detecion_out_channels
                    )
                ]
            )

    def forward(self, x: Tensor) -> list[Tensor]:
        out_list = []
        x = self.conv(x)
        x = self.blocks[0](x)
        x = self.blocks[1](x)

        out_list.append(x)
        x = self.blocks[2](x)
        out_list.append(x)
        x = self.blocks[3](x)
        out_list.append(x)
        x = self.blocks[4](x)
        out_list.append(x)

        if self.use_detection_backbone:
            out_list[0] = self.detecion_blocks[0](out_list[0])
            out_list[1] = self.detecion_blocks[1](out_list[1])
            out_list[2] = self.detecion_blocks[2](out_list[2])
            out_list[3] = self.detecion_blocks[3](out_list[3])
            return out_list

        x = F.adaptive_avg_pool2d(x, (1, self.max_text_len))

        return [x]
