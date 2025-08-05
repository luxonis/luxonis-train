from collections import defaultdict
from typing import Literal

from loguru import logger
from torch import Tensor, nn
from torch.utils import checkpoint

from luxonis_train.nodes.base_node import BaseNode
from luxonis_train.nodes.blocks import RepVGGBlock

from .variants import get_variant


class RepVGG(BaseNode[Tensor, list[Tensor]]):
    in_channels: int
    attach_index: int = -1

    def __init__(
        self,
        variant: Literal["A0", "A1", "A2"] = "A0",
        n_blocks: tuple[int, int, int, int] | None = None,
        width_multiplier: tuple[float, float, float, float] | None = None,
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


        @type variant: Literal["A0", "A1", "A2"]
        @param variant: RepVGG model variant. Defaults to "A0".
        @type n_blocks: tuple[int, int, int, int] | None
        @param n_blocks: Number of blocks in each stage.
        @type width_multiplier: tuple[float, float, float, float] | None
        @param width_multiplier: Width multiplier for each stage.
        @type override_groups_map: dict[int, int] | None
        @param override_groups_map: Dictionary mapping layer index to number of groups. The layers are indexed starting from 0.
        @type use_se: bool
        @param use_se: Whether to use Squeeze-and-Excitation blocks.
        @type use_checkpoint: bool
        @param use_checkpoint: Whether to use checkpointing.
        """
        super().__init__(**kwargs)
        var = get_variant(variant)

        n_blocks = n_blocks or var.n_blocks
        width_multiplier = width_multiplier or var.width_multiplier
        override_groups_map = defaultdict(lambda: 1, override_groups_map or {})
        self.use_se = use_se
        self.use_checkpoint = use_checkpoint

        self.in_planes = min(64, int(64 * width_multiplier[0]))
        self.stage0 = RepVGGBlock(
            in_channels=self.in_channels,
            out_channels=self.in_planes,
            kernel_size=3,
            stride=2,
            padding=1,
            use_se=self.use_se,
        )
        self.blocks = nn.ModuleList(
            [
                block
                for i in range(4)
                for block in self._make_stage(
                    int(2**i * 64 * width_multiplier[i]),
                    n_blocks[i],
                    stride=2,
                    groups=override_groups_map[i],
                )
            ]
        )
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)

    def forward(self, inputs: Tensor) -> list[Tensor]:
        outputs: list[Tensor] = []
        out = self.stage0(inputs)
        for block in self.blocks:
            if self.use_checkpoint:
                out = checkpoint.checkpoint(block, out)
            else:
                out = block(out)
            outputs.append(out)  # type: ignore
        return outputs

    def _make_stage(
        self, channels: int, n_blocks: int, stride: int, groups: int
    ) -> nn.ModuleList:
        strides = [stride] + [1] * (n_blocks - 1)
        blocks: list[nn.Module] = []
        for stride in strides:
            blocks.append(
                RepVGGBlock(
                    in_channels=self.in_planes,
                    out_channels=channels,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    groups=groups,
                    use_se=self.use_se,
                )
            )
            self.in_planes = channels
        return nn.ModuleList(blocks)

    def set_export_mode(self, mode: bool = True) -> None:
        """Reparametrizes instances of L{RepVGGBlock} in the network.

        @type mode: bool
        @param mode: Whether to set the export mode. Defaults to
            C{True}.
        """
        super().set_export_mode(mode)
        if self.export:
            logger.info("Reparametrizing RepVGG.")
            for module in self.modules():
                if isinstance(module, RepVGGBlock):
                    module.reparametrize()
