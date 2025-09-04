from luxonis_ml.typing import Kwargs
from torch import Tensor, nn
from typeguard import typechecked
from typing_extensions import override

from luxonis_train.nodes.base_node import BaseNode
from luxonis_train.nodes.blocks import (
    GeneralReparametrizableBlock,
    SqueezeExciteBlock,
)
from luxonis_train.nodes.blocks.utils import forward_gather


class MobileOne(BaseNode):
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

    Source: U{<https://github.com/apple/ml-mobileone>}

    Variants
    ========
    Each variant specifies a predefined set of values for:
      - width multipliers - A tuple of 4 float values specifying the width multipliers for each stage of the network. If the use of SE blocks is disabled, the last two values are ignored.
      - number of convolution branches - An integer specifying the number of linear convolution branches in MobileOne block.
      - use of SE blocks - A boolean specifying whether to use SE blocks in the network.

    Available variants are:
        - s0 (default): width_multipliers=(0.75, 1.0, 1.0, 2.0), n_conv_branches=4, use_se=False
        - s1: width_multipliers=(1.5, 1.5, 2.0, 2.5)
        - s2: width_multipliers=(1.5, 2.0, 2.5, 4.0)
        - s3: width_multipliers=(2.0, 2.5, 3.0, 4.0)
        - s4: width_multipliers=(3.0, 3.5, 3.5, 4.0), use_se=True

    @license: U{Apple<https://github.com/apple/ml-mobileone/blob/main/LICENSE>}
    """

    in_channels: int

    @typechecked
    def __init__(
        self,
        width_multipliers: tuple[float, float, float, float] = (
            0.75,
            1.0,
            1.0,
            2.0,
        ),
        n_conv_branches: int = 4,
        use_se: bool = False,
        **kwargs,
    ):
        """
        @type width_multipliers: tuple[float, float, float, float]
        @param width_multipliers: Width multipliers for each stage.
        @type n_conv_branches: int
        @param n_conv_branches: Number of linear convolution branches in MobileOne block.
        @type use_se: bool
        @param use_se: Whether to use C{Squeeze-and-Excitation} blocks in the network. Default is C{False}.
        """
        super().__init__(**kwargs)

        self.n_blocks_per_stage = [2, 8, 10, 1]
        self.n_conv_branches = n_conv_branches

        self._in_channels = min(64, int(64 * width_multipliers[0]))

        self.stages = nn.ModuleList(
            [
                GeneralReparametrizableBlock(
                    in_channels=self.in_channels,
                    out_channels=self._in_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                self._make_stage(
                    int(64 * width_multipliers[0]),
                    self.n_blocks_per_stage[0],
                    n_se_blocks=0,
                ),
                self._make_stage(
                    int(128 * width_multipliers[1]),
                    self.n_blocks_per_stage[1],
                    n_se_blocks=0,
                ),
                self._make_stage(
                    int(256 * width_multipliers[2]),
                    self.n_blocks_per_stage[2],
                    n_se_blocks=self.n_blocks_per_stage[2] // 2
                    if use_se
                    else 0,
                ),
                self._make_stage(
                    int(512 * width_multipliers[3]),
                    self.n_blocks_per_stage[3],
                    n_se_blocks=self.n_blocks_per_stage[3] if use_se else 0,
                ),
            ]
        )

    def forward(self, inputs: Tensor) -> list[Tensor]:
        return forward_gather(inputs, self.stages)

    @override
    @staticmethod
    def get_variants() -> tuple[str, dict[str, Kwargs]]:
        return "s0", {
            "s0": {
                "width_multipliers": (0.75, 1.0, 1.0, 2.0),
                "n_conv_branches": 4,
                "use_se": False,
            },
            "s1": {
                "width_multipliers": (1.5, 1.5, 2.0, 2.5),
                "use_se": False,
            },
            "s2": {
                "width_multipliers": (1.5, 2.0, 2.5, 4.0),
                "use_se": False,
            },
            "s3": {
                "width_multipliers": (2.0, 2.5, 3.0, 4.0),
                "use_se": False,
            },
            "s4": {
                "width_multipliers": (3.0, 3.5, 3.5, 4.0),
                "use_se": True,
            },
        }

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
                    n_branches=self.n_conv_branches,
                    refine_block=refine_block,
                    scale_layer_padding=0,
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
                    n_branches=self.n_conv_branches,
                    refine_block=refine_block,
                    use_scale_layer=False,
                )
            )
            self._in_channels = out_channels
        return nn.Sequential(*blocks)
