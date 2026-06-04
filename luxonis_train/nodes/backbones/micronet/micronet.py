from typing import TypedDict

from torch import Tensor, nn
from typeguard import typechecked
from typing_extensions import override

from luxonis_train.nodes.base_node import BaseNode

from .blocks import MicroBlock, Stem


class MicroNet(BaseNode):
    # TODO: Check docs, add source
    """MicroNet backbone.

    MicroNet is a lightweight convolutional backbone built from a stem and
    Dynamic Shift-Max MicroBlock stages.

    Metadata:
        - Node type: backbone
        - Registry name: ``MicroNet``
        - Task: None
        - Attach index: ``-1``
        - Inputs: ``features`` tensor
        - Outputs: ``features`` list of tensors

    Provenance:
        - Source: Unknown
        - License: Unknown
        - Implementation notes: Local MicroNet implementation with predefined
          stem and MicroBlock layer schedules.

    Variants:
        - ``"M1"``:
            - Default: yes
            - Aliases: None
            - Parameters:
                - ``stem_channels``: ``6``
                - ``stem_groups``: ``(3, 2)``
                - ``init_a``: ``(1.0, 1.0)``
                - ``init_b``: ``(0.0, 0.0)``
                - ``out_indices``: ``[1, 2, 4, 7]``
            - Layers:
                - ``0``:
                    - ``out_channels``: ``8``
                    - ``stride``: ``2``
                    - ``kernel_size``: ``3``
                    - ``expand_ratio``: ``(2, 2)``
                    - ``groups_1``: ``(0, 6)``
                    - ``groups_2``: ``(2, 2)``
                    - ``dy_shift``: ``(2, 0, 1)``
                    - ``reduction_factor``: ``1``
                - ``1``:
                    - ``out_channels``: ``16``
                    - ``stride``: ``2``
                    - ``kernel_size``: ``3``
                    - ``expand_ratio``: ``(2, 2)``
                    - ``groups_1``: ``(0, 8)``
                    - ``groups_2``: ``(4, 4)``
                    - ``dy_shift``: ``(2, 2, 1)``
                    - ``reduction_factor``: ``1``
                - ``2``:
                    - ``out_channels``: ``16``
                    - ``stride``: ``2``
                    - ``kernel_size``: ``5``
                    - ``expand_ratio``: ``(2, 2)``
                    - ``groups_1``: ``(0, 16)``
                    - ``groups_2``: ``(4, 4)``
                    - ``dy_shift``: ``(2, 2, 1)``
                    - ``reduction_factor``: ``1``
                - ``3``:
                    - ``out_channels``: ``32``
                    - ``stride``: ``1``
                    - ``kernel_size``: ``5``
                    - ``expand_ratio``: ``(1, 6)``
                    - ``groups_1``: ``(4, 4)``
                    - ``groups_2``: ``(4, 4)``
                    - ``dy_shift``: ``(2, 2, 1)``
                    - ``reduction_factor``: ``1``
                - ``4``:
                    - ``out_channels``: ``64``
                    - ``stride``: ``2``
                    - ``kernel_size``: ``5``
                    - ``expand_ratio``: ``(1, 6)``
                    - ``groups_1``: ``(8, 8)``
                    - ``groups_2``: ``(8, 8)``
                    - ``dy_shift``: ``(2, 2, 1)``
                    - ``reduction_factor``: ``1``
                - ``5``:
                    - ``out_channels``: ``96``
                    - ``stride``: ``1``
                    - ``kernel_size``: ``3``
                    - ``expand_ratio``: ``(1, 6)``
                    - ``groups_1``: ``(8, 8)``
                    - ``groups_2``: ``(8, 8)``
                    - ``dy_shift``: ``(2, 2, 1)``
                    - ``reduction_factor``: ``2``
                - ``6``:
                    - ``out_channels``: ``576``
                    - ``stride``: ``1``
                    - ``kernel_size``: ``3``
                    - ``expand_ratio``: ``(1, 6)``
                    - ``groups_1``: ``(12, 12)``
                    - ``groups_2``: ``(0, 0)``
                    - ``dy_shift``: ``(2, 2, 1)``
                    - ``reduction_factor``: ``2``
        - ``"M2"``:
            - Default: no
            - Aliases: None
            - Parameters:
                - ``stem_channels``: ``8``
                - ``stem_groups``: ``(4, 2)``
                - ``init_a``: ``(1.0, 1.0)``
                - ``init_b``: ``(0.0, 0.0)``
                - ``out_indices``: ``[1, 3, 6, 9]``
            - Layers:
                - ``0``:
                    - ``out_channels``: ``12``
                    - ``stride``: ``2``
                    - ``kernel_size``: ``3``
                    - ``expand_ratio``: ``(2, 2)``
                    - ``groups_1``: ``(0, 8)``
                    - ``groups_2``: ``(4, 4)``
                    - ``dy_shift``: ``(2, 0, 1)``
                    - ``reduction_factor``: ``1``
                - ``1``:
                    - ``out_channels``: ``16``
                    - ``stride``: ``2``
                    - ``kernel_size``: ``3``
                    - ``expand_ratio``: ``(2, 2)``
                    - ``groups_1``: ``(0, 12)``
                    - ``groups_2``: ``(4, 4)``
                    - ``dy_shift``: ``(2, 2, 1)``
                    - ``reduction_factor``: ``1``
                - ``2``:
                    - ``out_channels``: ``24``
                    - ``stride``: ``1``
                    - ``kernel_size``: ``3``
                    - ``expand_ratio``: ``(2, 2)``
                    - ``groups_1``: ``(0, 16)``
                    - ``groups_2``: ``(4, 4)``
                    - ``dy_shift``: ``(2, 2, 1)``
                    - ``reduction_factor``: ``1``
                - ``3``:
                    - ``out_channels``: ``32``
                    - ``stride``: ``2``
                    - ``kernel_size``: ``5``
                    - ``expand_ratio``: ``(1, 6)``
                    - ``groups_1``: ``(6, 6)``
                    - ``groups_2``: ``(4, 4)``
                    - ``dy_shift``: ``(2, 2, 1)``
                    - ``reduction_factor``: ``1``
                - ``4``:
                    - ``out_channels``: ``32``
                    - ``stride``: ``1``
                    - ``kernel_size``: ``5``
                    - ``expand_ratio``: ``(1, 6)``
                    - ``groups_1``: ``(8, 8)``
                    - ``groups_2``: ``(4, 4)``
                    - ``dy_shift``: ``(2, 2, 1)``
                    - ``reduction_factor``: ``2``
                - ``5``:
                    - ``out_channels``: ``64``
                    - ``stride``: ``1``
                    - ``kernel_size``: ``5``
                    - ``expand_ratio``: ``(1, 6)``
                    - ``groups_1``: ``(8, 8)``
                    - ``groups_2``: ``(8, 8)``
                    - ``dy_shift``: ``(2, 2, 1)``
                    - ``reduction_factor``: ``2``
                - ``6``:
                    - ``out_channels``: ``96``
                    - ``stride``: ``2``
                    - ``kernel_size``: ``5``
                    - ``expand_ratio``: ``(1, 6)``
                    - ``groups_1``: ``(8, 8)``
                    - ``groups_2``: ``(8, 8)``
                    - ``dy_shift``: ``(2, 2, 1)``
                    - ``reduction_factor``: ``2``
                - ``7``:
                    - ``out_channels``: ``128``
                    - ``stride``: ``1``
                    - ``kernel_size``: ``3``
                    - ``expand_ratio``: ``(1, 6)``
                    - ``groups_1``: ``(12, 12)``
                    - ``groups_2``: ``(8, 8)``
                    - ``dy_shift``: ``(2, 2, 1)``
                    - ``reduction_factor``: ``2``
                - ``8``:
                    - ``out_channels``: ``768``
                    - ``stride``: ``1``
                    - ``kernel_size``: ``3``
                    - ``expand_ratio``: ``(1, 6)``
                    - ``groups_1``: ``(16, 16)``
                    - ``groups_2``: ``(0, 0)``
                    - ``dy_shift``: ``(2, 2, 1)``
                    - ``reduction_factor``: ``2``
        - ``"M3"``:
            - Default: no
            - Aliases: None
            - Parameters:
                - ``stem_channels``: ``12``
                - ``stem_groups``: ``(4, 3)``
                - ``init_a``: ``(1.0, 0.5)``
                - ``init_b``: ``(0.0, 0.5)``
                - ``out_indices``: ``[1, 3, 8, 12]``
            - Layers:
                - ``0``:
                    - ``out_channels``: ``16``
                    - ``stride``: ``2``
                    - ``kernel_size``: ``3``
                    - ``expand_ratio``: ``(2, 2)``
                    - ``groups_1``: ``(0, 12)``
                    - ``groups_2``: ``(4, 4)``
                    - ``dy_shift``: ``(0, 2, 0)``
                    - ``reduction_factor``: ``1``
                - ``1``:
                    - ``out_channels``: ``24``
                    - ``stride``: ``2``
                    - ``kernel_size``: ``3``
                    - ``expand_ratio``: ``(2, 2)``
                    - ``groups_1``: ``(0, 16)``
                    - ``groups_2``: ``(4, 4)``
                    - ``dy_shift``: ``(0, 2, 0)``
                    - ``reduction_factor``: ``1``
                - ``2``:
                    - ``out_channels``: ``24``
                    - ``stride``: ``1``
                    - ``kernel_size``: ``3``
                    - ``expand_ratio``: ``(2, 2)``
                    - ``groups_1``: ``(0, 24)``
                    - ``groups_2``: ``(4, 4)``
                    - ``dy_shift``: ``(0, 2, 0)``
                    - ``reduction_factor``: ``1``
                - ``3``:
                    - ``out_channels``: ``32``
                    - ``stride``: ``2``
                    - ``kernel_size``: ``5``
                    - ``expand_ratio``: ``(1, 6)``
                    - ``groups_1``: ``(6, 6)``
                    - ``groups_2``: ``(4, 4)``
                    - ``dy_shift``: ``(0, 2, 0)``
                    - ``reduction_factor``: ``1``
                - ``4``:
                    - ``out_channels``: ``32``
                    - ``stride``: ``1``
                    - ``kernel_size``: ``5``
                    - ``expand_ratio``: ``(1, 6)``
                    - ``groups_1``: ``(8, 8)``
                    - ``groups_2``: ``(4, 4)``
                    - ``dy_shift``: ``(0, 2, 0)``
                    - ``reduction_factor``: ``2``
                - ``5``:
                    - ``out_channels``: ``64``
                    - ``stride``: ``1``
                    - ``kernel_size``: ``5``
                    - ``expand_ratio``: ``(1, 6)``
                    - ``groups_1``: ``(8, 8)``
                    - ``groups_2``: ``(8, 8)``
                    - ``dy_shift``: ``(0, 2, 0)``
                    - ``reduction_factor``: ``2``
                - ``6``:
                    - ``out_channels``: ``80``
                    - ``stride``: ``2``
                    - ``kernel_size``: ``5``
                    - ``expand_ratio``: ``(1, 6)``
                    - ``groups_1``: ``(8, 8)``
                    - ``groups_2``: ``(8, 8)``
                    - ``dy_shift``: ``(0, 2, 0)``
                    - ``reduction_factor``: ``2``
                - ``7``:
                    - ``out_channels``: ``80``
                    - ``stride``: ``1``
                    - ``kernel_size``: ``5``
                    - ``expand_ratio``: ``(1, 6)``
                    - ``groups_1``: ``(10, 10)``
                    - ``groups_2``: ``(8, 8)``
                    - ``dy_shift``: ``(0, 2, 0)``
                    - ``reduction_factor``: ``2``
                - ``8``:
                    - ``out_channels``: ``120``
                    - ``stride``: ``1``
                    - ``kernel_size``: ``5``
                    - ``expand_ratio``: ``(1, 6)``
                    - ``groups_1``: ``(10, 10)``
                    - ``groups_2``: ``(10, 10)``
                    - ``dy_shift``: ``(0, 2, 0)``
                    - ``reduction_factor``: ``2``
                - ``9``:
                    - ``out_channels``: ``120``
                    - ``stride``: ``1``
                    - ``kernel_size``: ``5``
                    - ``expand_ratio``: ``(1, 6)``
                    - ``groups_1``: ``(12, 12)``
                    - ``groups_2``: ``(10, 10)``
                    - ``dy_shift``: ``(0, 2, 0)``
                    - ``reduction_factor``: ``2``
                - ``10``:
                    - ``out_channels``: ``144``
                    - ``stride``: ``1``
                    - ``kernel_size``: ``3``
                    - ``expand_ratio``: ``(1, 6)``
                    - ``groups_1``: ``(12, 12)``
                    - ``groups_2``: ``(12, 12)``
                    - ``dy_shift``: ``(0, 2, 0)``
                    - ``reduction_factor``: ``2``
                - ``11``:
                    - ``out_channels``: ``864``
                    - ``stride``: ``1``
                    - ``kernel_size``: ``3``
                    - ``expand_ratio``: ``(1, 6)``
                    - ``groups_1``: ``(12, 12)``
                    - ``groups_2``: ``(0, 0)``
                    - ``dy_shift``: ``(0, 2, 0)``
                    - ``reduction_factor``: ``2``

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

        Args:
            stem_channels (int): Number of channels produced by the stem. Defaults to 6.
            stem_groups (tuple[int, int]): Channel grouping used by the stem. Defaults to (3, 2).
            init_a (tuple[float, float]): Initialization parameters for Dynamic Shift-Max. Defaults to (1.0, 1.0).
            init_b (tuple[float, float]): Initialization parameters for Dynamic Shift-Max. Defaults to (0.0, 0.0).
            out_indices (list[int] | None): Indices of the output layers. If provided, overrides the variant value.
            layer_params (list[LayerParamsDict] | None): Parameters for each MicroBlock. If provided, overrides the variant value.
            **kwargs (Any): Keyword arguments forwarded to the parent class.

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
