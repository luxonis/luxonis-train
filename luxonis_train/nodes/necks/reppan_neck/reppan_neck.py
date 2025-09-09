from typing import Literal

from loguru import logger
from luxonis_ml.typing import Kwargs
from torch import Tensor, nn
from typing_extensions import override

from luxonis_train.nodes.base_node import BaseNode
from luxonis_train.utils import make_divisible
from luxonis_train.variants import add_variant_aliases

from .blocks import CSPDownBlock, CSPUpBlock, RepDownBlock, RepUpBlock


class RepPANNeck(BaseNode):
    """Implementation of the RepPANNeck module. Supports the version
    with RepBlock and CSPStackRepBlock (for larger networks)

    Adapted from U{YOLOv6: A Single-Stage Object Detection Framework
    for Industrial Applications<https://arxiv.org/pdf/2209.02976.pdf>}.
    It has the balance of feature fusion ability and hardware efficiency.

    Variants
    ========
    The variant determines the depth and width multipliers, block used and intermediate channel scaling factor.
    Available variants:
        - "n" or "nano" (default): depth_multiplier=0.33, width_multiplier=0.25, block=RepBlock
        - "s" or "small": depth_multiplier=0.33, width_multiplier=0.50, block=RepBlock
        - "m" or "medium": depth_multiplier=0.60, width_multiplier=0.75, block=CSPStackRepBlock, e=2/3
        - "l" or "large": depth_multiplier=1.0, width_multiplier=1.0, block=CSPStackRepBlock, e=1/2
    """

    in_channels: list[int]
    in_width: list[int]
    in_height: list[int]

    def __init__(
        self,
        n_heads: Literal[2, 3, 4] = 3,
        channels_list: list[int] | None = None,
        n_repeats: list[int] | None = None,
        depth_multiplier: float = 0.33,
        width_multiplier: float = 0.25,
        block: Literal["RepBlock", "CSPStackRepBlock"] = "RepBlock",
        e: float | None = None,
        weights: str = "yolo",
        **kwargs,
    ):
        """
        @type n_heads: Literal[2,3,4]
        @param n_heads: Number of output heads. Defaults to 3. B{Note: Should be same
            also on head in most cases.}
        @type channels_list: list[int] | None
        @param channels_list: List of number of channels for each block.
            Defaults to C{[256, 128, 128, 256, 256, 512]}.
        @type n_repeats: list[int] | None
        @param n_repeats: List of number of repeats of RepVGGBlock.
            Defaults to C{[12, 12, 12, 12]}.
        @type depth_multiplier: float
        @param depth_multiplier: Depth multiplier. Defaults to C{0.33} ("n" variant).
        @type width_multiplier: float
        @param width_muliplier: Width multiplier. Defaults to C{0.25} ("n" variant).
        @type block: Literal["RepBlock", "CSPStackRepBlock"]
        @param block: Base block used when building the backbone.
            Defaults to C{"RepBlock"} ("n" variant).
        @tpe e: float | None
        @param e: Factor that controls number of intermediate channels.
            Only used when block="CSPStackRepBlock". Defaults to C{None}.
        """

        super().__init__(weights=weights, **kwargs)

        if (
            self.original_in_shape[-1] % 32 != 0
            or self.original_in_shape[-2] % 32 != 0
        ):
            logger.warning(
                "Image dimensions should be divisible by 32,"
                f"but got {self.original_in_shape}. "
                "This may cause 'RepPANNeck' to crash."
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
                    f"Expected width and height of feature map at index "
                    f"{len(self.in_width) - i} to be half of those at "
                    f"index {len(self.in_width) - i - 1}. "
                    f"Image shape {self.original_in_shape} must be "
                    "divisible by 32 to avoid 'RepPANNeck' crashing."
                )

        self.n_heads = n_heads

        channels_list = channels_list or [256, 128, 128, 256, 256, 512]
        n_repeats = n_repeats or [12, 12, 12, 12]
        channels_list = [
            make_divisible(ch * width_multiplier, 8) for ch in channels_list
        ]
        n_repeats = [
            (max(round(i * depth_multiplier), 1) if i > 1 else i)
            for i in n_repeats
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
                    e=e or 0.5,
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
                    e=e or 0.5,
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
            self.down_blocks, reversed(up_block_outs), strict=True
        ):
            x = down_block(x, up_out)
            outs.append(x)
        return outs

    @override
    def get_weights_url(self) -> str:
        return "{github}/reppanneck_{variant}_coco.ckpt"

    @override
    @staticmethod
    def get_variants() -> tuple[str, dict[str, Kwargs]]:
        return "n", add_variant_aliases(
            {
                "n": {
                    "depth_multiplier": 0.33,
                    "width_multiplier": 0.25,
                    "block": "RepBlock",
                },
                "s": {
                    "depth_multiplier": 0.33,
                    "width_multiplier": 0.50,
                    "block": "RepBlock",
                },
                "m": {
                    "depth_multiplier": 0.60,
                    "width_multiplier": 0.75,
                    "block": "CSPStackRepBlock",
                    "e": 2 / 3,
                },
                "l": {
                    "depth_multiplier": 1.0,
                    "width_multiplier": 1.0,
                    "block": "CSPStackRepBlock",
                    "e": 1 / 2,
                },
            }
        )

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
