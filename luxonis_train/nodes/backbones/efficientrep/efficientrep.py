import logging
from typing import Any

from torch import Tensor, nn

from luxonis_train.nodes.base_node import BaseNode
from luxonis_train.nodes.blocks import (
    BlockRepeater,
    RepVGGBlock,
    SpatialPyramidPoolingBlock,
)
from luxonis_train.utils import make_divisible

from .variants import VariantLiteral, get_variant

logger = logging.getLogger(__name__)


class EfficientRep(BaseNode[Tensor, list[Tensor]]):
    in_channels: int

    def __init__(
        self,
        variant: VariantLiteral = "nano",
        channels_list: list[int] | None = None,
        num_repeats: list[int] | None = None,
        depth_mul: float | None = None,
        width_mul: float | None = None,
        **kwargs: Any,
    ):
        """Implementation of the EfficientRep backbone.

        Adapted from U{YOLOv6: A Single-Stage Object Detection Framework
        for Industrial Applications
        <https://arxiv.org/pdf/2209.02976.pdf>}.

        @type variant: Literal["n", "nano", "s", "small", "m", "medium", "l", "large"]
        @param variant: EfficientRep variant. Defaults to "nano".
            The variant determines the depth and width multipliers.
            The depth multiplier determines the number of blocks in each stage and the width multiplier determines the number of channels.
            The following variants are available:
                - "n" or "nano" (default): depth_multiplier=0.33, width_multiplier=0.25
                - "s" or "small": depth_multiplier=0.33, width_multiplier=0.50
                - "m" or "medium": depth_multiplier=0.60, width_multiplier=0.75
                - "l" or "large": depth_multiplier=1.0, width_multiplier=1.0
        @type channels_list: list[int] | None
        @param channels_list: List of number of channels for each block. If unspecified,
            defaults to [64, 128, 256, 512, 1024].
        @type num_repeats: list[int] | None
        @param num_repeats: List of number of repeats of RepVGGBlock. If unspecified,
            defaults to [1, 6, 12, 18, 6].
        @type depth_mul: float
        @param depth_mul: Depth multiplier. If provided, overrides the variant value.
        @type width_mul: float
        @param width_mul: Width multiplier. If provided, overrides the variant value.
        """
        super().__init__(**kwargs)

        var = get_variant(variant)
        depth_mul = depth_mul or var.depth_multiplier
        width_mul = width_mul or var.width_multiplier

        channels_list = channels_list or [64, 128, 256, 512, 1024]
        num_repeats = num_repeats or [1, 6, 12, 18, 6]
        channels_list = [make_divisible(i * width_mul, 8) for i in channels_list]
        num_repeats = [
            (max(round(i * depth_mul), 1) if i > 1 else i) for i in num_repeats
        ]

        self.repvgg_encoder = RepVGGBlock(
            in_channels=self.in_channels,
            out_channels=channels_list[0],
            kernel_size=3,
            stride=2,
        )

        self.blocks = nn.ModuleList()
        for i in range(4):
            curr_block = nn.Sequential(
                RepVGGBlock(
                    in_channels=channels_list[i],
                    out_channels=channels_list[i + 1],
                    kernel_size=3,
                    stride=2,
                ),
                BlockRepeater(
                    block=RepVGGBlock,
                    in_channels=channels_list[i + 1],
                    out_channels=channels_list[i + 1],
                    num_blocks=num_repeats[i + 1],
                ),
            )
            self.blocks.append(curr_block)

        self.blocks[-1].append(
            SpatialPyramidPoolingBlock(
                in_channels=channels_list[4],
                out_channels=channels_list[4],
                kernel_size=5,
            )
        )

    def set_export_mode(self, mode: bool = True) -> None:
        """Reparametrizes instances of L{RepVGGBlock} in the network.

        @type mode: bool
        @param mode: Whether to set the export mode. Defaults to C{True}.
        """
        super().set_export_mode(mode)
        if self.export:
            logger.info("Reparametrizing 'EfficientRep'.")
            for module in self.modules():
                if isinstance(module, RepVGGBlock):
                    module.reparametrize()

    def forward(self, inputs: Tensor) -> list[Tensor]:
        outputs: list[Tensor] = []
        x = self.repvgg_encoder(inputs)
        for block in self.blocks:
            x = block(x)
            outputs.append(x)
        return outputs
