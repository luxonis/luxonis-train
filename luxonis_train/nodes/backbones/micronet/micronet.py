from typing import Literal

from torch import Tensor, nn

from luxonis_train.nodes.base_node import BaseNode

from .blocks import MicroBlock, Stem
from .variants import get_variant


class MicroNet(BaseNode[Tensor, list[Tensor]]):
    def __init__(
        self,
        variant: Literal["M1", "M2", "M3"] = "M1",
        out_indices: list[int] | None = None,
        **kwargs,
    ):
        """MicroNet backbone.

        This class creates the full MicroNet architecture based on the
        specified variant. It consists of a stem layer followed by
        multiple MicroBlocks.

        @type variant: Literal["M1", "M2", "M3"]
        @param variant: Model variant to use. Defaults to "M1".
        @type out_indices: list[int] | None
        @param out_indices: Indices of the output layers. If provided,
            overrides the variant value.
        """
        super().__init__(**kwargs)

        var = get_variant(variant)
        self.out_indices = out_indices or var.out_indices
        in_channels = var.stem_channels

        self.layers = nn.ModuleList([Stem(3, 2, var.stem_groups)])

        for bc in var.block_configs:
            self.layers.append(
                MicroBlock(
                    in_channels,
                    bc.out_channels,
                    bc.kernel_size,
                    bc.stride,
                    bc.expand_ratio,
                    bc.groups_1,
                    bc.groups_2,
                    bc.dy_shifts,
                    bc.reduction_factor,
                    var.init_a,
                    var.init_b,
                )
            )
            in_channels = bc.out_channels

    def forward(self, inputs: Tensor) -> list[Tensor]:
        outs: list[Tensor] = []
        for i, layer in enumerate(self.layers):
            inputs = layer(inputs)
            if i in self.out_indices:
                outs.append(inputs)
        return outs
