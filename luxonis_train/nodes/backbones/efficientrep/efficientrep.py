from typing import Literal

from loguru import logger
from torch import Tensor, nn

from luxonis_train.nodes.base_node import BaseNode
from luxonis_train.nodes.blocks import (
    BlockRepeater,
    CSPStackRepBlock,
    RepVGGBlock,
    SpatialPyramidPoolingBlock,
)
from luxonis_train.utils import make_divisible

from .variants import VariantLiteral, get_variant, get_variant_weights


class EfficientRep(BaseNode[Tensor, list[Tensor]]):
    in_channels: int

    def __init__(
        self,
        variant: VariantLiteral = "nano",
        channels_list: list[int] | None = None,
        n_repeats: list[int] | None = None,
        depth_mul: float | None = None,
        width_mul: float | None = None,
        block: Literal["RepBlock", "CSPStackRepBlock"] | None = None,
        csp_e: float | None = None,
        download_weights: bool = True,
        initialize_weights: bool = True,
        **kwargs,
    ):
        """Implementation of the EfficientRep backbone. Supports the
        version with RepBlock and CSPStackRepBlock (for larger networks)

        Adapted from U{YOLOv6: A Single-Stage Object Detection Framework
        for Industrial Applications
        <https://arxiv.org/pdf/2209.02976.pdf>}.

        @type variant: Literal["n", "nano", "s", "small", "m", "medium", "l", "large"]
        @param variant: EfficientRep variant. Defaults to "nano".
            The variant determines the depth and width multipliers, block used and intermediate channel scaling factor.
            The depth multiplier determines the number of blocks in each stage and the width multiplier determines the number of channels.
            The following variants are available:
                - "n" or "nano" (default): depth_multiplier=0.33, width_multiplier=0.25, block=RepBlock, e=None
                - "s" or "small": depth_multiplier=0.33, width_multiplier=0.50, block=RepBlock, e=None
                - "m" or "medium": depth_multiplier=0.60, width_multiplier=0.75, block=CSPStackRepBlock, e=2/3
                - "l" or "large": depth_multiplier=1.0, width_multiplier=1.0, block=CSPStackRepBlock, e=1/2
        @type channels_list: list[int] | None
        @param channels_list: List of number of channels for each block. If unspecified,
            defaults to [64, 128, 256, 512, 1024].
        @type n_repeats: list[int] | None
        @param n_repeats: List of number of repeats of RepVGGBlock. If unspecified,
            defaults to [1, 6, 12, 18, 6].
        @type depth_mul: float
        @param depth_mul: Depth multiplier. If provided, overrides the variant value.
        @type width_mul: float
        @param width_mul: Width multiplier. If provided, overrides the variant value.
        @type block: Literal["RepBlock", "CSPStackRepBlock"] | None
        @param block: Base block used when building the backbone. If provided, overrides the variant value.
        @type csp_e: float | None
        @param csp_e: Factor that controls number of intermediate channels if block="CSPStackRepBlock". If provided,
            overrides the variant value.
        @type download_weights: bool
        @param download_weights: If True download weights from COCO (if available for specified variant). Defaults to True.
        @type initialize_weights: bool
        @param initialize_weights: If True, initialize weights of the model.
        """
        super().__init__(**kwargs)

        var = get_variant(variant)
        depth_mul = depth_mul or var.depth_multiplier
        width_mul = width_mul or var.width_multiplier
        block = block or var.block
        csp_e = csp_e or var.csp_e or 0.5

        channels_list = channels_list or [64, 128, 256, 512, 1024]
        n_repeats = n_repeats or [1, 6, 12, 18, 6]
        channels_list = [
            make_divisible(i * width_mul, 8) for i in channels_list
        ]
        n_repeats = [
            (max(round(i * depth_mul), 1) if i > 1 else i) for i in n_repeats
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
                (
                    BlockRepeater(
                        block=RepVGGBlock,
                        in_channels=channels_list[i + 1],
                        out_channels=channels_list[i + 1],
                        n_blocks=n_repeats[i + 1],
                    )
                    if block == "RepBlock"
                    else CSPStackRepBlock(
                        in_channels=channels_list[i + 1],
                        out_channels=channels_list[i + 1],
                        n_blocks=n_repeats[i + 1],
                        e=csp_e,
                    )
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

        if initialize_weights:
            self.initialize_weights()

        if download_weights:
            weights_path = get_variant_weights(variant, initialize_weights)
            if weights_path:
                self.load_checkpoint(path=weights_path)
            else:
                logger.warning(
                    f"No checkpoint available for {self.name}, skipping."
                )

    def initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                pass
            elif isinstance(m, nn.BatchNorm2d):
                m.eps = 0.001
                m.momentum = 0.03
            elif isinstance(
                m, nn.Hardswish | nn.LeakyReLU | nn.ReLU | nn.ReLU6 | nn.SiLU
            ):
                m.inplace = True

    def set_export_mode(self, mode: bool = True) -> None:
        """Reparametrizes instances of L{RepVGGBlock} in the network.

        @type mode: bool
        @param mode: Whether to set the export mode. Defaults to
            C{True}.
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
