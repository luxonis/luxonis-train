from luxonis_ml.typing import Kwargs
from torch import Tensor, nn
from typeguard import typechecked
from typing_extensions import override

from luxonis_train.nodes.base_node import BaseNode

from .blocks import MicroBlock, Stem


class MicroNet(BaseNode[Tensor, list[Tensor]]):
    # TODO: Check docs, add source
    """MicroNet backbone.

    Variants
    --------
    The variant determines the architecture of the MicroNet backbone.
    Available variants are:
      - M1 (default):
        - stem_channels: 6
        - stem_groups: (3, 2)
        - init_a: (1.0, 1.0)
        - init_b: (0.0, 0.0)
        - out_indices: [1, 2, 4, 7]
        - strides: [2, 2, 2, 1, 2, 1, 1]
        - out_channels: [8, 16, 16, 32, 64, 96, 576]
        - kernel_sizes: [3, 3, 5, 5, 5, 3, 3]
        - expand_ratios: [(2, 2), (2, 2), (2, 2), (1, 6), (1, 6), (1, 6), (1, 6)]
        - groups_1: [(0, 6), (0, 8), (0, 16), (4, 4), (8, 8), (8, 8), (12, 12)]
        - groups_2: [(2, 2), (4, 4), (4, 4), (4, 4), (8, 9), (8, 8), (0, 0)]
        - dy_shifts: [(2, 0, 1), (2, 2, 1), (2, 2, 1), (2, 2, 1), (2, 2, 1), (2, 2, 1), (2, 2, 1)]
        - reduction_factors: [1, 1, 1, 1, 1, 2, 2]
      - M2:
        - stem_channels: 8
        - stem_groups: (4, 2)
        - init_a: (1.0, 1.0)
        - init_b: (0.0, 0.0)
        - out_indices: [1, 3, 6, 9]
        - strides: [2, 2, 1, 2, 1, 1, 2, 1, 1]
        - out_channels: [12, 16, 24, 32, 32, 64, 96, 128, 768]
        - kernel_sizes: [3, 3, 3, 5, 5, 5, 5, 3, 3]
        - expand_ratios: [(2, 2), (2, 2), (2, 2), (1, 6), ...]
        - groups_1: [(0, 8), (0, 12), (0, 16), (6, 6), (8, 8), (8, 8), (8, 8), (12, 12), (16, 16)]
        - groups_2: [(4, 4), (4, 4), (4, 4), (4, 4), (8, 8), (8, 8), (8, 8), (0, 0)]
        - dy_shifts: [(2, 0, 1), (2, 2, 1), ...]
        - reduction_factors: [1, 1, 1, 1, 2, 2, 2, 2, 2]
      - M3:
        - stem_channels: 12
        - stem_groups: (4, 3)
        - init_a: (1.0, 0.5)
        - init_b: (0.0, 0.5)
        - out_indices: [1, 3, 8, 12]
        - strides: [2, 2, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1]
        - out_channels: [16, 24, 24, 32, 32, 64, 80, 80, 120, 120, 144, 864]
        - kernel_sizes: [3, 3, 3, 5, 5, 5, 5, 5, 5, 5, 3, 3]
        - expand_ratios: [(2, 2), (2, 2), (2, 2), (1, 6), ...]
        - groups_1: [(0, 12), (0, 16), (0, 24), (6, 6), (8, 8), (8, 8), (8, 8), (10, 10), (10, 10), (12, 12), (12, 12), (12, 12)]
        - groups_2: [(4, 4), (4, 4), (4, 4), (4, 4), (4, 4), (8, 8), (8, 8), (8, 8), (10, 10), (10, 10), (12, 12), (0, 0)]
        - dy_shifts: [(0, 2, 0), ...]
        - reduction_factors: [1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2]
    """

    default_variant = "M1"

    @typechecked
    def __init__(
        self,
        stem_channels: int,
        stem_groups: tuple[int, int],
        init_a: tuple[float, float],
        init_b: tuple[float, float],
        out_indices: list[int],
        strides: list[int],
        out_channels: list[int],
        kernel_sizes: list[int],
        expand_ratios: list[tuple[int, int]],
        groups_1: list[tuple[int, int]],
        groups_2: list[tuple[int, int]],
        dy_shifts: list[tuple[int, int, int]],
        reduction_factors: list[int],
        **kwargs,
    ):
        """MicroNet backbone.

        This class creates the full MicroNet architecture based on the
        specified variant. It consists of a stem layer followed by
        multiple MicroBlocks.

        @type out_indices: list[int] | None
        @param out_indices: Indices of the output layers. If provided,
            overrides the variant value.
        """
        super().__init__(**kwargs)

        self.out_indices = out_indices
        self.layers = nn.ModuleList([Stem(3, 2, stem_groups)])

        in_channels = stem_channels
        for (
            out_channel,
            kernel_size,
            stride,
            expand_ratio,
            group_1,
            group_2,
            dy_shift,
            reduction_factor,
        ) in zip(
            out_channels,
            kernel_sizes,
            strides,
            expand_ratios,
            groups_1,
            groups_2,
            dy_shifts,
            reduction_factors,
            strict=True,
        ):
            self.layers.append(
                MicroBlock(
                    in_channels,
                    out_channel,
                    kernel_size,
                    stride,
                    expand_ratio,
                    group_1,
                    group_2,
                    dy_shift,
                    reduction_factor,
                    init_a,
                    init_b,
                )
            )
            in_channels = out_channel

    def forward(self, inputs: Tensor) -> list[Tensor]:
        outs: list[Tensor] = []
        for i, layer in enumerate(self.layers):
            inputs = layer(inputs)
            if i in self.out_indices:
                outs.append(inputs)
        return outs

    @override
    @staticmethod
    def get_variants() -> dict[str, Kwargs]:
        return {
            "M1": {
                "stem_channels": 6,
                "stem_groups": (3, 2),
                "init_a": (1.0, 1.0),
                "init_b": (0.0, 0.0),
                "out_indices": [1, 2, 4, 7],
                "strides": [2, 2, 2, 1, 2, 1, 1],
                "out_channels": [8, 16, 16, 32, 64, 96, 576],
                "kernel_sizes": [3, 3, 5, 5, 5, 3, 3],
                "expand_ratios": [(2, 2)] * 3 + [(1, 6)] * 4,
                "groups_1": [
                    (0, 6),
                    (0, 8),
                    (0, 16),
                    (4, 4),
                    (8, 8),
                    (8, 8),
                    (12, 12),
                ],
                "groups_2": [
                    (2, 2),
                    (4, 4),
                    (4, 4),
                    (4, 4),
                    (8, 8),
                    (8, 8),
                    (0, 0),
                ],
                "dy_shifts": [(2, 0, 1)] + [(2, 2, 1)] * 6,
                "reduction_factors": [1, 1, 1, 1, 1, 2, 2],
            },
            "M2": {
                "stem_channels": 8,
                "stem_groups": (4, 2),
                "init_a": (1.0, 1.0),
                "init_b": (0.0, 0.0),
                "out_indices": [1, 3, 6, 9],
                "strides": [2, 2, 1, 2, 1, 1, 2, 1, 1],
                "out_channels": [12, 16, 24, 32, 32, 64, 96, 128, 768],
                "kernel_sizes": [3, 3, 3, 5, 5, 5, 5, 3, 3],
                "expand_ratios": [(2, 2)] * 3 + [(1, 6)] * 6,
                "groups_1": [
                    (0, 8),
                    (0, 12),
                    (0, 16),
                    (6, 6),
                    (8, 8),
                    (8, 8),
                    (8, 8),
                    (12, 12),
                    (16, 16),
                ],
                "groups_2": [(4, 4)] * 5 + [(8, 8)] * 3 + [(0, 0)],
                "dy_shifts": [(2, 0, 1)] + [(2, 2, 1)] * 9,
                "reduction_factors": [1, 1, 1, 1, 2, 2, 2, 2, 2],
            },
            "M3": {
                "stem_channels": 12,
                "stem_groups": (4, 3),
                "init_a": (1.0, 0.5),
                "init_b": (0.0, 0.5),
                "out_indices": [1, 3, 8, 12],
                "strides": [2, 2, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1],
                "out_channels": [
                    16,
                    24,
                    24,
                    32,
                    32,
                    64,
                    80,
                    80,
                    120,
                    120,
                    144,
                    864,
                ],
                "kernel_sizes": [3, 3, 3, 5, 5, 5, 5, 5, 5, 5, 3, 3],
                "expand_ratios": [(2, 2)] * 3 + [(1, 6)] * 9,
                "groups_1": [
                    (0, 12),
                    (0, 16),
                    (0, 24),
                    (6, 6),
                    (8, 8),
                    (8, 8),
                    (8, 8),
                    (10, 10),
                    (10, 10),
                    (12, 12),
                    (12, 12),
                    (12, 12),
                ],
                "groups_2": [(4, 4)] * 5
                + [(8, 8)] * 3
                + [(10, 10), (10, 10), (12, 12), (0, 0)],
                "dy_shifts": [(0, 2, 0)] * 12,
                "reduction_factors": [1] * 4 + [2] * 8,
            },
        }
