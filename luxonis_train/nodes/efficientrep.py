"""Implementation of the EfficientRep backbone.

Adapted from U{YOLOv6: A Single-Stage Object Detection Framework for Industrial
Applications<https://arxiv.org/pdf/2209.02976.pdf>}.
"""

import logging

from torch import Tensor, nn

from luxonis_train.nodes.blocks import (
    BlockRepeater,
    RepVGGBlock,
    SpatialPyramidPoolingBlock,
)
from luxonis_train.utils.general import make_divisible

from .base_node import BaseNode


class EfficientRep(BaseNode[Tensor, list[Tensor]]):
    attach_index: int = -1

    def __init__(
        self,
        channels_list: list[int] | None = None,
        num_repeats: list[int] | None = None,
        depth_mul: float = 0.33,
        width_mul: float = 0.25,
        **kwargs,
    ):
        """EfficientRep backbone.

        @type channels_list: list[int] | None
        @param channels_list: List of number of channels for each block. Defaults to
            C{[64, 128, 256, 512, 1024]}.
        @type num_repeats: list[int] | None
        @param num_repeats: List of number of repeats of RepVGGBlock. Defaults to C{[1,
            6, 12, 18, 6]}.
        @type depth_mul: float
        @param depth_mul: Depth multiplier. Defaults to 0.33.
        @type width_mul: float
        @param width_mul: Width multiplier. Defaults to 0.25.
        @type kwargs: Any
        @param kwargs: Additional arguments to pass to L{BaseNode}.
        """
        super().__init__(**kwargs)

        channels_list = channels_list or [64, 128, 256, 512, 1024]
        num_repeats = num_repeats or [1, 6, 12, 18, 6]
        channels_list = [make_divisible(i * width_mul, 8) for i in channels_list]
        num_repeats = [
            (max(round(i * depth_mul), 1) if i > 1 else i) for i in num_repeats
        ]

        in_channels = self.in_channels
        if not isinstance(in_channels, int):
            raise ValueError("EfficientRep module expects only one input.")

        self.repvgg_encoder = RepVGGBlock(
            in_channels=in_channels,
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
        """Reparametrizes instances of `RepVGGBlock` in the network.

        @type mode: bool
        @param mode: Whether to set the export mode. Defaults to C{True}.
        """
        super().set_export_mode(mode)
        logger = logging.getLogger(__name__)
        if mode:
            logger.info("Reparametrizing EfficientRep.")
            for module in self.modules():
                if isinstance(module, RepVGGBlock):
                    module.reparametrize()

    def forward(self, x: Tensor) -> list[Tensor]:
        outputs = []
        x = self.repvgg_encoder(x)
        for block in self.blocks:
            x = block(x)
            outputs.append(x)
        return outputs
