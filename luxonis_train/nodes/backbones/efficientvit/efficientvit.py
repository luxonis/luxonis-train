from torch import Tensor, nn

from luxonis_train.nodes.base_node import BaseNode
from luxonis_train.nodes.blocks import ConvModule

from .blocks import (
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
        **kwargs,
    ):
        """EfficientViT backbone implementation based on a lightweight
        transformer architecture.

        This implementation is inspired by the architecture described in the paper:
        "EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction"
        (https://arxiv.org/abs/2205.14756).

        The EfficientViT model is designed to provide a balance between computational efficiency
        and performance, making it suitable for deployment on edge devices with limited resources.

        @type variant: Literal["n", "nano", "s", "small", "m", "medium", "l", "large"]
        @param variant: EfficientViT variant. Defaults to "nano".
            The variant determines the width, depth, and dimension of the network. The following variants are available:
                - "n" or "nano" (default): width_list=[8, 16, 32, 64, 128], depth_list=[1, 2, 2, 2, 2], dim=16
                - "s" or "small": width_list=[16, 32, 64, 128, 256], depth_list=[1, 2, 3, 3, 4], dim=16
                - "m" or "medium": width_list=[24, 48, 96, 192, 384], depth_list=[1, 3, 4, 4, 6], dim=32
                - "l" or "large": width_list=[32, 64, 128, 256, 512], depth_list=[1, 4, 6, 6, 9], dim=32

        @type width_list: list[int] | None
        @param width_list: List of number of channels for each block. If unspecified, defaults to the variant's width_list.
        @type depth_list: list[int] | None
        @param depth_list: List of number of layers in each block. If unspecified, defaults to the variant's depth_list.
        @type expand_ratio: int
        @param expand_ratio: Expansion ratio for the MobileBottleneckBlock. Defaults to 4.
        @type dim: int | None
        @param dim: Dimension of the transformer. Defaults to the variant's dim.
        """
        super().__init__(**kwargs)

        var = get_variant(variant)
        width_list = width_list or var.width_list
        depth_list = depth_list or var.depth_list
        dim = dim or var.dim

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
            for block in encoder_blocks:  # type: ignore
                x = block(x)
            outputs.append(x)
        return outputs
