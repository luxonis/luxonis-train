from luxonis_ml.typing import Kwargs
from torch import Tensor, nn
from typeguard import typechecked
from typing_extensions import override

from luxonis_train.nodes.base_node import BaseNode
from luxonis_train.nodes.blocks import ConvBlock
from luxonis_train.variants import add_variant_aliases

from .blocks import (
    DepthWiseSeparableConv,
    EfficientViTBlock,
    MobileBottleneckBlock,
)


class EfficientViT(BaseNode):
    """EfficientViT backbone implementation based on a lightweight
    transformer architecture.

    This implementation is inspired by the architecture described in the paper:
    "EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction"
    (https://arxiv.org/abs/2205.14756).

    The EfficientViT model is designed to provide a balance between computational efficiency
    and performance, making it suitable for deployment on edge devices with limited resources.

    Variants
    ========
    The variant determines the width, depth, and dimension of the network.
    Available variants are:
      - "n" or "nano" (default): width_list=[8, 16, 32, 64, 128], depth_list=[1, 2, 2, 2, 2], dim=16
      - "s" or "small": width_list=[16, 32, 64, 128, 256], depth_list=[1, 2, 3, 3, 4], dim=16
      - "m" or "medium": width_list=[24, 48, 96, 192, 384], depth_list=[1, 3, 4, 4, 6], dim=32
      - "l" or "large": width_list=[32, 64, 128, 256, 512], depth_list=[1, 4, 6, 6, 9], dim=32
    """

    in_channels: int

    @typechecked
    def __init__(
        self,
        width_list: list[int] | None = None,
        depth_list: list[int] | None = None,
        dim: int = 16,
        expand_ratio: int = 4,
        **kwargs,
    ):
        """
        @type width_list: list[int]
        @param width_list: List of number of channels for each block.
        @type depth_list: list[int]
        @param depth_list: List of number of layers in each block.
        @type dim: int | None
        @param dim: Dimension of the transformer.
        @type expand_ratio: int
        @param expand_ratio: Expansion ratio for the L{MobileBottleneckBlock}. Defaults to C{4}.
        """
        super().__init__(**kwargs)
        width_list = width_list or [8, 16, 32, 64, 128]
        depth_list = depth_list or [1, 2, 2, 2, 2]

        self.feature_extractor = nn.ModuleList(
            [
                ConvBlock(
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
            block = DepthWiseSeparableConv(
                in_channels=width_list[0],
                out_channels=width_list[0],
                stride=1,
                depthwise_activation=nn.Hardswish(),
                use_residual=True,
            )
            self.feature_extractor.append(block)

        in_channels = width_list[0]
        self.encoder_blocks = nn.ModuleList()
        for w, d in zip(width_list[1:3], depth_list[1:3], strict=True):
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
                    use_residual=stride == 1,
                )
                encoder_blocks.append(block)
                in_channels = w
            self.encoder_blocks.append(encoder_blocks)

        for w, d in zip(width_list[3:], depth_list[3:], strict=True):
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
                        n_channels=in_channels,
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
            for block in encoder_blocks:  # type: ignore
                x = block(x)
            outputs.append(x)
        return outputs

    @override
    @staticmethod
    def get_variants() -> tuple[str, dict[str, Kwargs]]:
        return "n", add_variant_aliases(
            {
                "n": {
                    "width_list": [8, 16, 32, 64, 128],
                    "depth_list": [1, 2, 2, 2, 2],
                    "dim": 16,
                },
                "s": {
                    "width_list": [16, 32, 64, 128, 256],
                    "depth_list": [1, 2, 3, 3, 4],
                    "dim": 16,
                },
                "m": {
                    "width_list": [24, 48, 96, 192, 384],
                    "depth_list": [1, 3, 4, 4, 6],
                    "dim": 32,
                },
                "l": {
                    "width_list": [32, 64, 128, 256, 512],
                    "depth_list": [1, 4, 6, 6, 9],
                    "dim": 32,
                },
            }
        )
