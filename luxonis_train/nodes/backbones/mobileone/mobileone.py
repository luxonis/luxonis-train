"""MobileOne backbone.

Source: U{<https://github.com/apple/ml-mobileone>}
@license: U{Apple<https://github.com/apple/ml-mobileone/blob/main/LICENSE>}
"""

from typing import Literal

from torch import Tensor, nn

from luxonis_train.nodes.base_node import BaseNode
from luxonis_train.nodes.blocks import GeneralReparametrizableBlock
from luxonis_train.nodes.blocks.blocks import SqueezeExciteBlock

from .variants import get_variant


class MobileOne(BaseNode[Tensor, list[Tensor]]):
    in_channels: int

    def __init__(
        self,
        variant: Literal["s0", "s1", "s2", "s3", "s4"] = "s0",
        width_multipliers: tuple[float, float, float, float] | None = None,
        n_conv_branches: int | None = None,
        use_se: bool | None = None,
        **kwargs,
    ):
        """MobileOne: An efficient CNN backbone for mobile devices.

        The architecture focuses on reducing memory access costs and improving parallelism
        while allowing aggressive parameter scaling for better representation capacity.
        Different variants (S0-S4) offer various accuracy-latency tradeoffs.

        Key features:
            - Designed for low latency on mobile while maintaining high accuracy
            - Uses re-parameterizable branches during training that get folded at inference
            - Employs trivial over-parameterization branches for improved accuracy
            - Simple feed-forward structure at inference with no branches/skip connections
            - Variants achieve <1ms inference time on iPhone 12 with up to 75.9% top-1 ImageNet accuracy
            - Outperforms other efficient architectures like MobileNets on image classification,
              object detection and semantic segmentation tasks
            - Uses only basic operators available across platforms (no custom activations)


        Reference: U{MobileOne: An Improved One millisecond Mobile Backbone
        <https://arxiv.org/abs/2206.04040>}

        @type variant: Literal["s0", "s1", "s2", "s3", "s4"]
        @param variant: Specifies which variant of the MobileOne network to use. Defaults to "s0".
            Each variant specifies a predefined set of values for:
                - width multipliers - A tuple of 4 float values specifying the width multipliers for each stage of the network. If the use of SE blocks is disabled, the last two values are ignored.
                - number of convolution branches - An integer specifying the number of linear convolution branches in MobileOne block.
                - use of SE blocks - A boolean specifying whether to use SE blocks in the network.

            The variants are as follows:
                - s0 (default): width_multipliers=(0.75, 1.0, 1.0, 2.0), n_conv_branches=4, use_se=False
                - s1: width_multipliers=(1.5, 1.5, 2.0, 2.5), n_conv_branches=1, use_se=False
                - s2: width_multipliers=(1.5, 2.0, 2.5, 4.0), n_conv_branches=1, use_se=False
                - s3: width_multipliers=(2.0, 2.5, 3.0, 4.0), n_conv_branches=1, use_se=False
                - s4: width_multipliers=(3.0, 3.5, 3.5, 4.0), n_conv_branches=1, use_se=True

        @type width_multipliers: tuple[float, float, float, float] | None
        @param width_multipliers: Width multipliers for each stage. If provided, overrides the variant values.
        @type n_conv_branches: int | None
        @param n_conv_branches: Number of linear convolution branches in MobileOne block. If provided, overrides the variant values.
        @type use_se: bool | None
        @param use_se: Whether to use C{Squeeze-and-Excitation} blocks in the network. If provided, overrides the variant value.
        """
        super().__init__(**kwargs)

        var = get_variant(variant)

        width_multipliers = width_multipliers or var.width_multipliers
        use_se = use_se or var.use_se
        self.n_blocks_per_stage = [2, 8, 10, 1]
        self.n_conv_branches = n_conv_branches or var.n_conv_branches

        self._in_channels = min(64, int(64 * width_multipliers[0]))

        self.stage0 = GeneralReparametrizableBlock(
            in_channels=self.in_channels,
            out_channels=self._in_channels,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.cur_layer_idx = 1
        self.stage1 = self._make_stage(
            int(64 * width_multipliers[0]),
            self.n_blocks_per_stage[0],
            n_se_blocks=0,
        )
        self.stage2 = self._make_stage(
            int(128 * width_multipliers[1]),
            self.n_blocks_per_stage[1],
            n_se_blocks=0,
        )
        self.stage3 = self._make_stage(
            int(256 * width_multipliers[2]),
            self.n_blocks_per_stage[2],
            n_se_blocks=self.n_blocks_per_stage[2] // 2 if use_se else 0,
        )
        self.stage4 = self._make_stage(
            int(512 * width_multipliers[3]),
            self.n_blocks_per_stage[3],
            n_se_blocks=self.n_blocks_per_stage[3] if use_se else 0,
        )

    def forward(self, inputs: Tensor) -> list[Tensor]:
        outs: list[Tensor] = []
        x = self.stage0(inputs)
        outs.append(x)
        x = self.stage1(x)
        outs.append(x)
        x = self.stage2(x)
        outs.append(x)
        x = self.stage3(x)
        outs.append(x)
        x = self.stage4(x)
        outs.append(x)

        return outs

    def _make_stage(
        self, out_channels: int, n_blocks: int, n_se_blocks: int
    ) -> nn.Sequential:
        """Build a stage of MobileOne model.

        @type out_channels: int
        @param out_channels: Number of output channels.
        @type n_blocks: int
        @param n_blocks: Number of blocks in this stage.
        @type n_se_blocks: int
        @param n_se_blocks: Number of SE blocks in this stage.
        @rtype: nn.Sequential
        @return: A stage of MobileOne model.
        """
        # Get strides for all layers
        strides = [2] + [1] * (n_blocks - 1)
        blocks: list[nn.Module] = []
        for ix, stride in enumerate(strides):
            if n_se_blocks > n_blocks:
                raise ValueError(
                    "Number of SE blocks cannot exceed number of layers."
                )
            if ix >= (n_blocks - n_se_blocks):
                refine_block = SqueezeExciteBlock(
                    in_channels=self._in_channels,
                    intermediate_channels=self._in_channels // 16,
                )
            else:
                refine_block = nn.Identity()

            # Depthwise conv
            blocks.append(
                GeneralReparametrizableBlock(
                    in_channels=self._in_channels,
                    out_channels=self._in_channels,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    groups=self._in_channels,
                    num_branches=self.n_conv_branches,
                    refine_block=refine_block,
                )
            )
            # Pointwise conv
            blocks.append(
                GeneralReparametrizableBlock(
                    in_channels=self._in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    groups=1,
                    num_branches=self.n_conv_branches,
                    refine_block=refine_block,
                )
            )
            self._in_channels = out_channels
            self.cur_layer_idx += 1
        return nn.Sequential(*blocks)
