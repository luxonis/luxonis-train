from typing import Literal

from loguru import logger
from torch import Tensor, nn

from luxonis_train.nodes.base_node import BaseNode
from luxonis_train.nodes.blocks import RepVGGBlock
from luxonis_train.utils import make_divisible

from .blocks import CSPDownBlock, CSPUpBlock, RepDownBlock, RepUpBlock
from .variants import VariantLiteral, get_variant, get_variant_weights


class RepPANNeck(BaseNode[list[Tensor], list[Tensor]]):
    in_channels: list[int]
    in_width: list[int]
    in_height: list[int]

    def __init__(
        self,
        variant: VariantLiteral = "nano",
        n_heads: Literal[2, 3, 4] = 3,
        channels_list: list[int] | None = None,
        n_repeats: list[int] | None = None,
        depth_mul: float | None = None,
        width_mul: float | None = None,
        block: Literal["RepBlock", "CSPStackRepBlock"] | None = None,
        csp_e: float | None = None,
        download_weights: bool = False,
        initialize_weights: bool = True,
        **kwargs,
    ):
        """Implementation of the RepPANNeck module. Supports the version
        with RepBlock and CSPStackRepBlock (for larger networks)

        Adapted from U{YOLOv6: A Single-Stage Object Detection Framework
        for Industrial Applications<https://arxiv.org/pdf/2209.02976.pdf>}.
        It has the balance of feature fusion ability and hardware efficiency.

        @type variant: Literal["n", "nano", "s", "small", "m", "medium", "l", "large"]
        @param variant: RepPANNeck variant. Defaults to "nano".
            The variant determines the depth and width multipliers, block used and intermediate channel scaling factor.
            The depth multiplier determines the number of blocks in each stage and the width multiplier determines the number of channels.
            The following variants are available:
                - "n" or "nano" (default): depth_multiplier=0.33, width_multiplier=0.25, block=RepBlock, e=None
                - "s" or "small": depth_multiplier=0.33, width_multiplier=0.50, block=RepBlock, e=None
                - "m" or "medium": depth_multiplier=0.60, width_multiplier=0.75, block=CSPStackRepBlock, e=2/3
                - "l" or "large": depth_multiplier=1.0, width_multiplier=1.0, block=CSPStackRepBlock, e=1/2
        @type n_heads: Literal[2,3,4]
        @param n_heads: Number of output heads. Defaults to 3. B{Note: Should be same
            also on head in most cases.}
        @type channels_list: list[int] | None
        @param channels_list: List of number of channels for each block.
            Defaults to C{[256, 128, 128, 256, 256, 512]}.
        @type n_repeats: list[int] | None
        @param n_repeats: List of number of repeats of RepVGGBlock.
            Defaults to C{[12, 12, 12, 12]}.
        @type depth_mul: float
        @param depth_mul: Depth multiplier. Defaults to C{0.33}.
        @type width_mul: float
        @param width_mul: Width multiplier. Defaults to C{0.25}.
        @type block: Literal["RepBlock", "CSPStackRepBlock"] | None
        @param block: Base block used when building the backbone. If provided, overrides the variant value.
        @tpe csp_e: float | None
        @param csp_e: Factor that controls number of intermediate channels if block="CSPStackRepBlock". If provided,
            overrides the variant value.
        @type download_weights: bool
        @param download_weights: If True download weights from COCO (if available for specified variant). Defaults to False.
        @type initialize_weights: bool
        @param initialize_weights: If True, initialize weights of the model.
        """
        super().__init__(**kwargs)

        if (
            self.original_in_shape[-1] % 32 != 0
            or self.original_in_shape[-2] % 32 != 0
        ):
            logger.warning(
                "Image dimensions should be divisible by 32. This may cause 'RepPANNeck' to crash."
            )

        for i in range(1, n_heads):
            if (
                (
                    self.in_width[-i] * 2 != self.in_width[-i - 1]
                    or self.in_height[-i] * 2 != self.in_height[-i - 1]
                )
                and self.attach_index == "all"
            ):  # TODO: fix the attach_index and this condition
                raise ValueError(
                    f"Expected width and height of feature map at index {len(self.in_width) - i} to be half of those at index {len(self.in_width) - i - 1}. "
                    f"Image shape {self.original_in_shape} must be divisible by 32 to avoid 'RepPANNeck' crashing."
                )

        self.n_heads = n_heads

        var = get_variant(variant)
        depth_mul = depth_mul or var.depth_multiplier
        width_mul = width_mul or var.width_multiplier
        block = block or var.block
        csp_e = csp_e or var.csp_e or 0.5

        channels_list = channels_list or [256, 128, 128, 256, 256, 512]
        n_repeats = n_repeats or [12, 12, 12, 12]
        channels_list = [
            make_divisible(ch * width_mul, 8) for ch in channels_list
        ]
        n_repeats = [
            (max(round(i * depth_mul), 1) if i > 1 else i) for i in n_repeats
        ]

        channels_list, n_repeats = self._fit_to_n_heads(
            channels_list, n_repeats
        )

        self.up_blocks = nn.ModuleList()

        in_channels = self.in_channels[-1]
        out_channels = channels_list[0]
        in_channels_next = self.in_channels[-2]
        curr_n_repeats = n_repeats[0]
        up_out_channel_list = [in_channels]  # used in DownBlocks

        for i in range(1, n_heads):
            curr_up_block = (
                RepUpBlock(
                    in_channels=in_channels,
                    in_channels_next=in_channels_next,
                    out_channels=out_channels,
                    n_repeats=curr_n_repeats,
                )
                if block == "RepBlock"
                else CSPUpBlock(
                    in_channels=in_channels,
                    in_channels_next=in_channels_next,
                    out_channels=out_channels,
                    n_repeats=curr_n_repeats,
                    e=csp_e,
                )
            )
            up_out_channel_list.append(out_channels)
            self.up_blocks.append(curr_up_block)
            if len(self.up_blocks) == (n_heads - 1):
                up_out_channel_list.reverse()
                break

            in_channels = out_channels
            out_channels = channels_list[i]
            in_channels_next = self.in_channels[-1 - (i + 1)]
            curr_n_repeats = n_repeats[i]

        self.down_blocks = nn.ModuleList()
        channels_list_down_blocks = channels_list[(n_heads - 1) :]
        n_repeats_down_blocks = n_repeats[(n_heads - 1) :]

        in_channels = out_channels
        downsample_out_channels = channels_list_down_blocks[0]
        in_channels_next = up_out_channel_list[0]
        out_channels = channels_list_down_blocks[1]
        curr_n_repeats = n_repeats_down_blocks[0]

        for i in range(1, n_heads):
            curr_down_block = (
                RepDownBlock(
                    in_channels=in_channels,
                    downsample_out_channels=downsample_out_channels,
                    in_channels_next=in_channels_next,
                    out_channels=out_channels,
                    n_repeats=curr_n_repeats,
                )
                if block == "RepBlock"
                else CSPDownBlock(
                    in_channels=in_channels,
                    downsample_out_channels=downsample_out_channels,
                    in_channels_next=in_channels_next,
                    out_channels=out_channels,
                    n_repeats=curr_n_repeats,
                    e=csp_e,
                )
            )
            self.down_blocks.append(curr_down_block)
            if len(self.down_blocks) == (n_heads - 1):
                break

            in_channels = out_channels
            downsample_out_channels = channels_list_down_blocks[2 * i]
            in_channels_next = up_out_channel_list[i]
            out_channels = channels_list_down_blocks[2 * i + 1]
            curr_n_repeats = n_repeats_down_blocks[i]

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

    def forward(self, inputs: list[Tensor]) -> list[Tensor]:
        x = inputs[-1]
        up_block_outs: list[Tensor] = []
        for up_block, input_ in zip(
            self.up_blocks, inputs[-2::-1], strict=False
        ):
            conv_out, x = up_block(x, input_)
            up_block_outs.append(conv_out)

        outs = [x]
        for down_block, up_out in zip(
            self.down_blocks, reversed(up_block_outs)
        ):
            x = down_block(x, up_out)
            outs.append(x)
        return outs

    def set_export_mode(self, mode: bool = True) -> None:
        """Reparametrizes instances of L{RepVGGBlock} in the network.

        @type mode: bool
        @param mode: Whether to set the export mode. Defaults to
            C{True}.
        """
        super().set_export_mode(mode)
        if self.export:
            logger.info("Reparametrizing 'RepPANNeck'.")
            for module in self.modules():
                if isinstance(module, RepVGGBlock):
                    module.reparametrize()

    def _fit_to_n_heads(
        self, channels_list: list[int], n_repeats: list[int]
    ) -> tuple[list[int], list[int]]:
        """Fits channels_list and n_repeats to n_heads by removing or
        adding items.

        Also scales the numbers based on offset
        """
        if self.n_heads == 2:
            channels_list = [channels_list[i] for i in [0, 4, 5]]
            n_repeats = [n_repeats[0], n_repeats[3]]
        elif self.n_heads == 3:
            return channels_list, n_repeats
        elif self.n_heads == 4:
            channels_list = [
                channels_list[0],
                channels_list[1],
                channels_list[1] // 2,
                channels_list[1] // 2,
                channels_list[1],
                channels_list[2],
                channels_list[3],
                channels_list[4],
                channels_list[5],
            ]
            n_repeats = [n_repeats[i] for i in [0, 1, 1, 2, 2, 3]]
        else:
            raise ValueError(
                f"Specified number of heads ({self.n_heads}) not supported."
                "The number of heads should be 2, 3 or 4."
            )

        return channels_list, n_repeats
