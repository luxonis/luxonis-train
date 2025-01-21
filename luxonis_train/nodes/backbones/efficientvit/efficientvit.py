from typing import Any

import torch.nn as nn
from torch import Tensor

from luxonis_train.nodes.base_node import BaseNode
from luxonis_train.nodes.blocks import (
    ConvModule,
    DepthwiseSeparableConv,
    EfficientViTBlock,
    MobileBottleneckBlock,
)

from .variants import VariantLiteral, get_variant


class EfficientViT(BaseNode[Tensor, list[Tensor]]):
    in_channels: int

    def __init__(
        self,
        variant: VariantLiteral = "n",
        width_list: list[int] | None = None,
        depth_list: list[int] | None = None,
        expand_ratio: int = 4,
        dim: int | None = None,
        **kwargs: Any,
    ):
        """EfficientViT backbone implementation that is based on the
        transformer architecture.

        This implementation is inspired by the architecture described in the paper:
        U{EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction}
        <https://arxiv.org/abs/2205.14756>.

        The EfficientViT model leverages the transformer architecture to achieve a balance between
        computational efficiency and model performance. The model is designed to be lightweight and
        suitable for deployment on edge devices with limited resources.


        @type variant: Literal["n", "nano", "s", "small", "m", "medium", "l", "large"]
        @param variant: EfficientViT variant. Defaults to "nano".
            The variant determines the depth and width multipliers, block used and intermediate channel scaling factor.
            The depth multiplier determines the number of blocks in each stage and the width multiplier determines the number of channels.
            The following variants are available:
                - "n" or "nano" (default): depth_multiplier=0.33, width_multiplier=0.25, block=RepBlock, e=None
                - "s" or "small": depth_multiplier=0.33, width_multiplier=0.50, block=RepBlock, e=None
                - "m" or "medium": depth_multiplier=0.60, width_multiplier=0.75, block=CSPStackRepBlock, e=2/3
                - "l" or "large": depth_multiplier=1.0, width_multiplier=1.0, block=CSPStackRepBlock, e=1/2

        @type width_list: list[int] | None
        @param width_list: List of number of channels for each block. If unspecified,
            defaults to [64, 128, 256, 512, 1024].
        @type depth_list: list[int] | None
        @param depth_list: List of number of repeats of RepVGGBlock. If unspecified,
            defaults to [1, 6, 12, 18, 6].
        @type expand_ratio: int
        @param expand_ratio: Expansion ratio for the MobileBottleneckBlock. Defaults to 4.
        @type dim: int | None
        @param dim: Dimension of the transformer. Defaults to None.
        @type kwargs: Any
        @param kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)

        variant = get_variant(variant)
        width_list = width_list or variant.width_list
        depth_list = depth_list or variant.depth_list
        dim = dim or variant.dim

        # feature_extractor
        self.feature_extractor = nn.ModuleList(
            [
                ConvModule(
                    in_channels=self.in_channels,
                    out_channels=width_list[0],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    activation=nn.Hardswish(),
                )
            ]
        )
        for _ in range(depth_list[0]):
            block = DepthwiseSeparableConv(
                in_channels=width_list[0],
                out_channels=width_list[0],
                stride=1,
                activation=[nn.Hardswish(), nn.Identity()],
                use_residual=True,
                use_bias=[False, False],
            )
            self.feature_extractor.append(block)

        # encoder_blocks
        in_channels = width_list[0]
        self.encoder_blocks = nn.ModuleList()
        for w, d in zip(width_list[1:3], depth_list[1:3]):
            encoder_blocks = nn.ModuleList()
            for i in range(d):
                stride = 2 if i == 0 else 1
                block = MobileBottleneckBlock(
                    in_channels=in_channels,
                    out_channels=w,
                    stride=stride,
                    expand_ratio=expand_ratio,
                    use_norm=[True, True, True],
                    activation=[
                        nn.Hardswish(),
                        nn.Hardswish(),
                        nn.Identity(),
                    ],
                    use_bias=[False, False, False],
                    use_residual=True if stride == 1 else False,
                )
                encoder_blocks.append(block)
                in_channels = w
            self.encoder_blocks.append(encoder_blocks)

        for w, d in zip(width_list[3:], depth_list[3:]):
            encoder_blocks = nn.ModuleList()
            block = MobileBottleneckBlock(
                in_channels=in_channels,
                out_channels=w,
                stride=2,
                expand_ratio=expand_ratio,
                activation=[
                    nn.Hardswish(),
                    nn.Hardswish(),
                    nn.Identity(),
                ],
                use_norm=[False, False, True],
                use_residual=False,
            )
            encoder_blocks.append(block)
            in_channels = w

            for _ in range(d):
                encoder_blocks.append(
                    EfficientViTBlock(
                        num_channels=in_channels,
                        head_dim=dim,
                        expansion_factor=expand_ratio,
                    )
                )
            self.encoder_blocks.append(encoder_blocks)

    def forward(self, x: Tensor) -> list[Tensor]:
        outputs = []
        for block in self.feature_extractor:
            x = block(x)
        outputs.append(x)
        for encoder_blocks in self.encoder_blocks:
            for block in encoder_blocks:
                x = block(x)
            outputs.append(x)
        return outputs
