from typing import Literal, cast

from luxonis_ml.typing import Kwargs
from torch import Tensor, nn
from typing_extensions import override

from luxonis_train.nodes.base_node import BaseNode
from luxonis_train.nodes.blocks import (
    BlockRepeater,
    CSPStackRepBlock,
    GeneralReparametrizableBlock,
    SpatialPyramidPoolingBlock,
)
from luxonis_train.utils import make_divisible
from luxonis_train.variants import add_variant_aliases


class EfficientRep(BaseNode):
    """EfficientRep backbone for object detection.

    Variants
    ========
    The variant determines the depth and width multipliers,
    block used and intermediate channel scaling factor.

    The depth multiplier determines the number of blocks in each stage and the width multiplier determines the number of channels.

    The following variants are available:
      - "n" or "nano" (default): depth_multiplier=0.33, width_multiplier=0.25, block=RepBlock, e=None
      - "s" or "small": depth_multiplier=0.33, width_multiplier=0.50, block=RepBlock, e=None
      - "m" or "medium": depth_multiplier=0.60, width_multiplier=0.75, block=CSPStackRepBlock, e=2/3
      - "l" or "large": depth_multiplier=1.0, width_multiplier=1.0, block=CSPStackRepBlock, e=1/2
    """

    in_channels: int

    def __init__(
        self,
        channels_list: list[int] | None = None,
        n_repeats: list[int] | None = None,
        depth_multiplier: float = 0.33,
        width_multiplier: float = 0.25,
        block: Literal["RepBlock", "CSPStackRepBlock"] = "RepBlock",
        csp_e: float = 0.5,
        weights: str = "yolo",
        **kwargs,
    ):
        """Implementation of the EfficientRep backbone. Supports the
        version with RepBlock and CSPStackRepBlock (for larger networks)

        Adapted from U{YOLOv6: A Single-Stage Object Detection Framework
        for Industrial Applications
        <https://arxiv.org/pdf/2209.02976.pdf>}.

        @type channels_list: list[int] | None
        @param channels_list: List of number of channels for each block.
            If unspecified, defaults to [64, 128, 256, 512, 1024].
        @type n_repeats: list[int] | None
        @param n_repeats: List of number of repeats of RepVGGBlock. If
            unspecified, defaults to [1, 6, 12, 18, 6].
        @type depth_mul: float
        @param depth_mul: Depth multiplier. If provided, overrides the
            variant value.
        @type width_mul: float
        @param width_mul: Width multiplier. If provided, overrides the
            variant value.
        @type block: Literal["RepBlock", "CSPStackRepBlock"] | None
        @param block: Base block used when building the backbone. If
            provided, overrides the variant value.
        @type csp_e: float | None
        @param csp_e: Factor that controls number of intermediate
            channels if block="CSPStackRepBlock". If provided, overrides
            the variant value.
        """
        super().__init__(weights=weights, **kwargs)

        channels_list = channels_list or [64, 128, 256, 512, 1024]
        n_repeats = n_repeats or [1, 6, 12, 18, 6]
        channels_list = [
            make_divisible(i * width_multiplier, 8) for i in channels_list
        ]
        n_repeats = [
            (max(round(i * depth_multiplier), 1) if i > 1 else i)
            for i in n_repeats
        ]

        self.repvgg_encoder = GeneralReparametrizableBlock(
            in_channels=self.in_channels,
            out_channels=channels_list[0],
            kernel_size=3,
            stride=2,
        )

        self.blocks = cast(list[nn.Sequential], nn.ModuleList())
        for i in range(4):
            curr_block = nn.Sequential(
                GeneralReparametrizableBlock(
                    in_channels=channels_list[i],
                    out_channels=channels_list[i + 1],
                    kernel_size=3,
                    stride=2,
                ),
                (
                    BlockRepeater(
                        GeneralReparametrizableBlock,
                        in_channels=channels_list[i + 1],
                        out_channels=channels_list[i + 1],
                        n_repeats=n_repeats[i + 1],
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

    def forward(self, inputs: Tensor) -> list[Tensor]:
        outputs: list[Tensor] = []
        x = self.repvgg_encoder(inputs)
        for block in self.blocks:
            x = block(x)
            outputs.append(x)
        return outputs

    @override
    def get_weights_url(self) -> str:
        return "{github}/efficientrep_{variant}_coco.ckpt"

    @staticmethod
    @override
    def get_variants() -> tuple[str, dict[str, Kwargs]]:
        return "n", add_variant_aliases(
            {
                "n": {
                    "depth_multiplier": 0.33,
                    "width_multiplier": 0.25,
                    "block": "RepBlock",
                    "csp_e": None,
                },
                "s": {
                    "depth_multiplier": 0.33,
                    "width_multiplier": 0.50,
                    "block": "RepBlock",
                    "csp_e": None,
                },
                "m": {
                    "depth_multiplier": 0.60,
                    "width_multiplier": 0.75,
                    "block": "CSPStackRepBlock",
                    "csp_e": 2 / 3,
                },
                "l": {
                    "depth_multiplier": 1.0,
                    "width_multiplier": 1.0,
                    "block": "CSPStackRepBlock",
                    "csp_e": 1 / 2,
                },
            }
        )
