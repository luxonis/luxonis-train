from collections import defaultdict
from typing import cast

from luxonis_ml.typing import Kwargs
from torch import Tensor, nn
from torch.utils import checkpoint
from typeguard import typechecked
from typing_extensions import override

from luxonis_train.nodes.base_node import BaseNode
from luxonis_train.nodes.blocks import GeneralReparametrizableBlock


class RepVGG(BaseNode[Tensor, list[Tensor]]):
    """RepVGG backbone.

    Variants
    ========

    The variant determines the number of blocks in each stage and the width multiplier.

    The following variants are available:
        - "A0" (default): n_blocks=(2, 4, 14, 1), width_multiplier=(0.75, 0.75, 0.75, 2.5)
        - "A1": n_blocks=(2, 4, 14, 1), width_multiplier=(1, 1, 1, 2.5)
        - "A2": n_blocks=(2, 4, 14, 1), width_multiplier=(1.5, 1.5, 1.5, 2.75)
    """

    in_channels: int
    attach_index: int = -1

    @typechecked
    def __init__(
        self,
        n_blocks: tuple[int, int, int, int],
        width_multiplier: tuple[float, float, float, float],
        override_groups_map: dict[int, int] | None = None,
        use_se: bool = False,
        use_checkpoint: bool = False,
        **kwargs,
    ):
        """RepVGG backbone.

        RepVGG is a VGG-style convolutional architecture.

            - Simple feed-forward topology without any branching.
            - 3x3 convolutions and ReLU activations.
            - No automatic search, manual refinement or compound scaling.

        @license: U{MIT
            <https://github.com/DingXiaoH/RepVGG/blob/main/LICENSE>}.

        @see: U{https://github.com/DingXiaoH/RepVGG}
        @see: U{https://paperswithcode.com/method/repvgg}
        @see: U{RepVGG: Making VGG-style ConvNets Great Again
            <https://arxiv.org/abs/2101.03697>}


        @type n_blocks: tuple[int, int, int, int]
        @param n_blocks: Number of blocks in each stage.
        @type width_multiplier: tuple[float, float, float, float]
        @param width_multiplier: Width multiplier for each stage.
        @type override_groups_map: dict[int, int] | None
        @param override_groups_map: Dictionary mapping layer index to number of groups. The layers are indexed starting from 0.
        @type use_se: bool
        @param use_se: Whether to use Squeeze-and-Excitation blocks.
        @type use_checkpoint: bool
        @param use_checkpoint: Whether to use checkpointing.
        """
        super().__init__(**kwargs)

        override_groups_map = defaultdict(lambda: 1, override_groups_map or {})
        self.use_se = use_se
        self.use_checkpoint = use_checkpoint

        out_channels = min(64, int(64 * width_multiplier[0]))
        self.stage0 = GeneralReparametrizableBlock(
            in_channels=self.in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            refine_block="se",
        )

        blocks = []
        in_channels = out_channels
        for i in range(4):
            out_channels = int(2**i * 64 * width_multiplier[i])
            blocks.extend(
                self._make_stage(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    strides=[2] + [1] * (n_blocks[i] - 1),
                    groups=override_groups_map[i],
                )
            )
            in_channels = out_channels

        self.blocks = nn.ModuleList(blocks)
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)

    def forward(self, inputs: Tensor) -> list[Tensor]:
        outputs: list[Tensor] = []
        out = self.stage0(inputs)
        for block in self.blocks:
            # TODO: What exactly does this do?
            if self.use_checkpoint:
                out = cast(Tensor, checkpoint.checkpoint(block, out))
            else:
                out = block(out)
            outputs.append(out)
        return outputs

    @override
    @staticmethod
    def get_variants() -> tuple[str, dict[str, Kwargs]]:
        return "A0", {
            "A0": {
                "n_blocks": (2, 4, 14, 1),
                "width_multiplier": (0.75, 0.75, 0.75, 2.5),
            },
            "A1": {
                "n_blocks": (2, 4, 14, 1),
                "width_multiplier": (1, 1, 1, 2.5),
            },
            "A2": {
                "n_blocks": (2, 4, 14, 1),
                "width_multiplier": (1.5, 1.5, 1.5, 2.75),
            },
        }

    def _make_stage(
        self,
        in_channels: int,
        out_channels: int,
        strides: list[int],
        groups: int,
    ) -> list[nn.Module]:
        stage = []
        for stride in strides:
            stage.append(
                GeneralReparametrizableBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    groups=groups,
                    refine_block="se" if self.use_se else None,
                )
            )
            in_channels = out_channels
        return stage
