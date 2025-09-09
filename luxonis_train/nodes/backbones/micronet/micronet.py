from typing import TypedDict

from torch import Tensor, nn
from typeguard import typechecked
from typing_extensions import override

from luxonis_train.nodes.base_node import BaseNode

from .blocks import MicroBlock, Stem


class MicroNet(BaseNode):
    # TODO: Check docs, add source
    """MicroNet backbone.

    Variants
    ========
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

    @typechecked
    def __init__(
        self,
        stem_channels: int = 6,
        stem_groups: tuple[int, int] = (3, 2),
        init_a: tuple[float, float] = (1.0, 1.0),
        init_b: tuple[float, float] = (0.0, 0.0),
        out_indices: list[int] | None = None,
        layer_params: list["LayerParamsDict"] | None = None,
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
        out_indices = out_indices or [1, 2, 4, 7]
        layer_params = (
            layer_params or self.get_variants()[1]["M1"]["layer_params"]
        )

        self.out_indices = out_indices
        self.layers = nn.ModuleList([Stem(3, 2, stem_groups)])

        in_channels = stem_channels
        for params in layer_params:
            self.layers.append(
                MicroBlock(
                    in_channels,
                    init_a=init_a,
                    init_b=init_b,
                    **params,
                )
            )
            in_channels = params["out_channels"]

    def forward(self, inputs: Tensor) -> list[Tensor]:
        outs: list[Tensor] = []
        for i, layer in enumerate(self.layers):
            inputs = layer(inputs)
            if i in self.out_indices:
                outs.append(inputs)
        return outs

    @override
    @staticmethod
    def get_variants() -> tuple[str, dict[str, "MicroNetVariantDict"]]:
        return "M1", {
            "M1": {
                "stem_channels": 6,
                "stem_groups": (3, 2),
                "init_a": (1.0, 1.0),
                "init_b": (0.0, 0.0),
                "out_indices": [1, 2, 4, 7],
                "layer_params": [
                    {
                        "out_channels": 8,
                        "stride": 2,
                        "kernel_size": 3,
                        "expand_ratio": (2, 2),
                        "groups_1": (0, 6),
                        "groups_2": (2, 2),
                        "dy_shift": (2, 0, 1),
                        "reduction_factor": 1,
                    },
                    {
                        "out_channels": 16,
                        "stride": 2,
                        "kernel_size": 3,
                        "expand_ratio": (2, 2),
                        "groups_1": (0, 8),
                        "groups_2": (4, 4),
                        "dy_shift": (2, 2, 1),
                        "reduction_factor": 1,
                    },
                    {
                        "out_channels": 16,
                        "stride": 2,
                        "kernel_size": 5,
                        "expand_ratio": (2, 2),
                        "groups_1": (0, 16),
                        "groups_2": (4, 4),
                        "dy_shift": (2, 2, 1),
                        "reduction_factor": 1,
                    },
                    {
                        "out_channels": 32,
                        "stride": 1,
                        "kernel_size": 5,
                        "expand_ratio": (1, 6),
                        "groups_1": (4, 4),
                        "groups_2": (4, 4),
                        "dy_shift": (2, 2, 1),
                        "reduction_factor": 1,
                    },
                    {
                        "out_channels": 64,
                        "stride": 2,
                        "kernel_size": 5,
                        "expand_ratio": (1, 6),
                        "groups_1": (8, 8),
                        "groups_2": (8, 8),
                        "dy_shift": (2, 2, 1),
                        "reduction_factor": 1,
                    },
                    {
                        "out_channels": 96,
                        "stride": 1,
                        "kernel_size": 3,
                        "expand_ratio": (1, 6),
                        "groups_1": (8, 8),
                        "groups_2": (8, 8),
                        "dy_shift": (2, 2, 1),
                        "reduction_factor": 2,
                    },
                    {
                        "out_channels": 576,
                        "stride": 1,
                        "kernel_size": 3,
                        "expand_ratio": (1, 6),
                        "groups_1": (12, 12),
                        "groups_2": (0, 0),
                        "dy_shift": (2, 2, 1),
                        "reduction_factor": 2,
                    },
                ],
            },
            "M2": {
                "stem_channels": 8,
                "stem_groups": (4, 2),
                "init_a": (1.0, 1.0),
                "init_b": (0.0, 0.0),
                "out_indices": [1, 3, 6, 9],
                "layer_params": [
                    {
                        "out_channels": 12,
                        "stride": 2,
                        "kernel_size": 3,
                        "expand_ratio": (2, 2),
                        "groups_1": (0, 8),
                        "groups_2": (4, 4),
                        "dy_shift": (2, 0, 1),
                        "reduction_factor": 1,
                    },
                    {
                        "out_channels": 16,
                        "stride": 2,
                        "kernel_size": 3,
                        "expand_ratio": (2, 2),
                        "groups_1": (0, 12),
                        "groups_2": (4, 4),
                        "dy_shift": (2, 2, 1),
                        "reduction_factor": 1,
                    },
                    {
                        "out_channels": 24,
                        "stride": 1,
                        "kernel_size": 3,
                        "expand_ratio": (2, 2),
                        "groups_1": (0, 16),
                        "groups_2": (4, 4),
                        "dy_shift": (2, 2, 1),
                        "reduction_factor": 1,
                    },
                    {
                        "out_channels": 32,
                        "stride": 2,
                        "kernel_size": 5,
                        "expand_ratio": (1, 6),
                        "groups_1": (6, 6),
                        "groups_2": (4, 4),
                        "dy_shift": (2, 2, 1),
                        "reduction_factor": 1,
                    },
                    {
                        "out_channels": 32,
                        "stride": 1,
                        "kernel_size": 5,
                        "expand_ratio": (1, 6),
                        "groups_1": (8, 8),
                        "groups_2": (4, 4),
                        "dy_shift": (2, 2, 1),
                        "reduction_factor": 2,
                    },
                    {
                        "out_channels": 64,
                        "stride": 1,
                        "kernel_size": 5,
                        "expand_ratio": (1, 6),
                        "groups_1": (8, 8),
                        "groups_2": (8, 8),
                        "dy_shift": (2, 2, 1),
                        "reduction_factor": 2,
                    },
                    {
                        "out_channels": 96,
                        "stride": 2,
                        "kernel_size": 5,
                        "expand_ratio": (1, 6),
                        "groups_1": (8, 8),
                        "groups_2": (8, 8),
                        "dy_shift": (2, 2, 1),
                        "reduction_factor": 2,
                    },
                    {
                        "out_channels": 128,
                        "stride": 1,
                        "kernel_size": 3,
                        "expand_ratio": (1, 6),
                        "groups_1": (12, 12),
                        "groups_2": (8, 8),
                        "dy_shift": (2, 2, 1),
                        "reduction_factor": 2,
                    },
                    {
                        "out_channels": 768,
                        "stride": 1,
                        "kernel_size": 3,
                        "expand_ratio": (1, 6),
                        "groups_1": (16, 16),
                        "groups_2": (0, 0),
                        "dy_shift": (2, 2, 1),
                        "reduction_factor": 2,
                    },
                ],
            },
            "M3": {
                "stem_channels": 12,
                "stem_groups": (4, 3),
                "init_a": (1.0, 0.5),
                "init_b": (0.0, 0.5),
                "out_indices": [1, 3, 8, 12],
                "layer_params": [
                    {
                        "out_channels": 16,
                        "stride": 2,
                        "kernel_size": 3,
                        "expand_ratio": (2, 2),
                        "groups_1": (0, 12),
                        "groups_2": (4, 4),
                        "dy_shift": (0, 2, 0),
                        "reduction_factor": 1,
                    },
                    {
                        "out_channels": 24,
                        "stride": 2,
                        "kernel_size": 3,
                        "expand_ratio": (2, 2),
                        "groups_1": (0, 16),
                        "groups_2": (4, 4),
                        "dy_shift": (0, 2, 0),
                        "reduction_factor": 1,
                    },
                    {
                        "out_channels": 24,
                        "stride": 1,
                        "kernel_size": 3,
                        "expand_ratio": (2, 2),
                        "groups_1": (0, 24),
                        "groups_2": (4, 4),
                        "dy_shift": (0, 2, 0),
                        "reduction_factor": 1,
                    },
                    {
                        "out_channels": 32,
                        "stride": 2,
                        "kernel_size": 5,
                        "expand_ratio": (1, 6),
                        "groups_1": (6, 6),
                        "groups_2": (4, 4),
                        "dy_shift": (0, 2, 0),
                        "reduction_factor": 1,
                    },
                    {
                        "out_channels": 32,
                        "stride": 1,
                        "kernel_size": 5,
                        "expand_ratio": (1, 6),
                        "groups_1": (8, 8),
                        "groups_2": (4, 4),
                        "dy_shift": (0, 2, 0),
                        "reduction_factor": 2,
                    },
                    {
                        "out_channels": 64,
                        "stride": 1,
                        "kernel_size": 5,
                        "expand_ratio": (1, 6),
                        "groups_1": (8, 8),
                        "groups_2": (8, 8),
                        "dy_shift": (0, 2, 0),
                        "reduction_factor": 2,
                    },
                    {
                        "out_channels": 80,
                        "stride": 2,
                        "kernel_size": 5,
                        "expand_ratio": (1, 6),
                        "groups_1": (8, 8),
                        "groups_2": (8, 8),
                        "dy_shift": (0, 2, 0),
                        "reduction_factor": 2,
                    },
                    {
                        "out_channels": 80,
                        "stride": 1,
                        "kernel_size": 5,
                        "expand_ratio": (1, 6),
                        "groups_1": (10, 10),
                        "groups_2": (8, 8),
                        "dy_shift": (0, 2, 0),
                        "reduction_factor": 2,
                    },
                    {
                        "out_channels": 120,
                        "stride": 1,
                        "kernel_size": 5,
                        "expand_ratio": (1, 6),
                        "groups_1": (10, 10),
                        "groups_2": (10, 10),
                        "dy_shift": (0, 2, 0),
                        "reduction_factor": 2,
                    },
                    {
                        "out_channels": 120,
                        "stride": 1,
                        "kernel_size": 5,
                        "expand_ratio": (1, 6),
                        "groups_1": (12, 12),
                        "groups_2": (10, 10),
                        "dy_shift": (0, 2, 0),
                        "reduction_factor": 2,
                    },
                    {
                        "out_channels": 144,
                        "stride": 1,
                        "kernel_size": 3,
                        "expand_ratio": (1, 6),
                        "groups_1": (12, 12),
                        "groups_2": (12, 12),
                        "dy_shift": (0, 2, 0),
                        "reduction_factor": 2,
                    },
                    {
                        "out_channels": 864,
                        "stride": 1,
                        "kernel_size": 3,
                        "expand_ratio": (1, 6),
                        "groups_1": (12, 12),
                        "groups_2": (0, 0),
                        "dy_shift": (0, 2, 0),
                        "reduction_factor": 2,
                    },
                ],
            },
        }


class LayerParamsDict(TypedDict):
    out_channels: int
    stride: int
    kernel_size: int
    expand_ratio: tuple[int, int]
    groups_1: tuple[int, int]
    groups_2: tuple[int, int]
    dy_shift: tuple[int, int, int]
    reduction_factor: int


class MicroNetVariantDict(TypedDict):
    stem_channels: int
    stem_groups: tuple[int, int]
    init_a: tuple[float, float]
    init_b: tuple[float, float]
    out_indices: list[int]
    layer_params: list[LayerParamsDict]
