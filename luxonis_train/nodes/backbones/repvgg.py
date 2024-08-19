import logging
from typing import Literal

import torch.utils.checkpoint as checkpoint
from torch import Tensor, nn

from luxonis_train.nodes.blocks import RepVGGBlock

from ..base_node import BaseNode

logger = logging.getLogger(__name__)


class RepVGG(BaseNode):
    """Implementation of RepVGG backbone.

    Source: U{https://github.com/DingXiaoH/RepVGG}
    @license: U{MIT<https://github.com/DingXiaoH/RepVGG/blob/main/LICENSE>}.

    @todo: technical documentation
    """

    in_channels: int
    attach_index: int = -1

    VARIANTS_SETTINGS = {
        "A0": {
            "num_blocks": [2, 4, 14, 1],
            "width_multiplier": [0.75, 0.75, 0.75, 2.5],
        },
        "A1": {
            "num_blocks": [2, 4, 14, 1],
            "width_multiplier": [1, 1, 1, 2.5],
        },
        "A2": {
            "num_blocks": [2, 4, 14, 1],
            "width_multiplier": [1.5, 1.5, 1.5, 2.75],
        },
    }

    def __init__(
        self,
        variant: Literal["A0", "A1", "A2"] = "A0",
        num_blocks: list[int] | None = None,
        width_multiplier: list[float] | None = None,
        override_groups_map: dict[int, int] | None = None,
        use_se: bool = False,
        use_checkpoint: bool = False,
        **kwargs,
    ):
        """Constructor for the RepVGG module.

        @type variant: Literal["A0", "A1", "A2"]
        @param variant: RepVGG model variant. Defaults to "A0".
        @type override_groups_map: dict[int, int] | None
        @param override_groups_map: Dictionary mapping layer index to number of groups.
        @type use_se: bool
        @param use_se: Whether to use Squeeze-and-Excitation blocks.
        @type use_checkpoint: bool
        @param use_checkpoint: Whether to use checkpointing.
        @type num_blocks: list[int] | None
        @param num_blocks: Number of blocks in each stage.
        @type width_multiplier: list[float] | None
        @param width_multiplier: Width multiplier for each stage.
        """
        super().__init__(**kwargs)
        if variant not in self.VARIANTS_SETTINGS.keys():
            raise ValueError(
                f"RepVGG model variant should be one of "
                f"{list(self.VARIANTS_SETTINGS.keys())}."
            )

        num_blocks = num_blocks or self.VARIANTS_SETTINGS[variant]["num_blocks"]
        width_multiplier = (
            width_multiplier or self.VARIANTS_SETTINGS[variant]["width_multiplier"]
        )
        self.override_groups_map = override_groups_map or {}
        assert 0 not in self.override_groups_map
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
        self.cur_layer_idx = 1
        self.stage1 = self._make_stage(
            int(64 * width_multiplier[0]), num_blocks[0], stride=2
        )
        self.stage2 = self._make_stage(
            int(128 * width_multiplier[1]), num_blocks[1], stride=2
        )
        self.stage3 = self._make_stage(
            int(256 * width_multiplier[2]), num_blocks[2], stride=2
        )
        self.stage4 = self._make_stage(
            int(512 * width_multiplier[3]), num_blocks[3], stride=2
        )
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)

    def forward(self, inputs: Tensor) -> list[Tensor]:
        outputs = []
        out = self.stage0(inputs)
        for stage in (self.stage1, self.stage2, self.stage3, self.stage4):
            for block in stage:
                if self.use_checkpoint:
                    out = checkpoint.checkpoint(block, out)
                else:
                    out = block(out)
            outputs.append(out)
        return outputs

    def _make_stage(self, planes: int, num_blocks: int, stride: int):
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for stride in strides:
            cur_groups = self.override_groups_map.get(self.cur_layer_idx, 1)
            blocks.append(
                RepVGGBlock(
                    in_channels=self.in_planes,
                    out_channels=planes,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    groups=cur_groups,
                    use_se=self.use_se,
                )
            )
            self.in_planes = planes
            self.cur_layer_idx += 1
        return nn.ModuleList(blocks)

    def set_export_mode(self, mode: bool = True) -> None:
        """Reparametrizes instances of L{RepVGGBlock} in the network.

        @type mode: bool
        @param mode: Whether to set the export mode. Defaults to C{True}.
        """
        super().set_export_mode(mode)
        if self.export:
            logger.info("Reparametrizing RepVGG.")
            for module in self.modules():
                if isinstance(module, RepVGGBlock):
                    module.reparametrize()
