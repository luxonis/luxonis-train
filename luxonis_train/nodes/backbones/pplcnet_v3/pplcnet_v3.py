from typing import TypedDict

from torch import Tensor, nn
from typeguard import typechecked
from typing_extensions import override

from luxonis_train.nodes.base_node import BaseNode
from luxonis_train.nodes.blocks import ConvBlock

from .blocks import LCNetV3Layer, scale_up


class PPLCNetV3(BaseNode):
    """PPLCNetV3 backbone.

    PPLCNetV3 is a PaddleOCR-inspired lightweight convolutional backbone
    for OCR recognition and optional detection-style feature outputs.

    Metadata:
        - Node type: backbone
        - Registry name: ``PPLCNetV3``
        - Task: None
        - Attach index: ``-1``
        - Inputs: ``features`` tensor
        - Outputs: ``features`` list of tensors

    Provenance:
        - Source: ``PaddlePaddle/PaddleOCR``
        - License: Apache License, Version 2.0
        - Implementation notes: Local LCNetV3 layer implementation with
          recognition and detection-backbone output modes.

    Variants:
        - ``"rec-light"``:
            - Default: yes
            - Aliases: None
            - Parameters:
                - ``scale``: ``0.95``
                - ``n_branches``: ``4``
                - ``use_detection_backbone``: ``False``
            - Layers:
                - ``0``:
                    - ``kernel_sizes``: ``[3]``
                    - ``out_channels``: ``[32]``
                    - ``strides``: ``[1]``
                    - ``use_se``: ``[False]``
                - ``1``:
                    - ``kernel_sizes``: ``[3, 3]``
                    - ``out_channels``: ``[64, 64]``
                    - ``strides``: ``[2, 1]``
                    - ``use_se``: ``[False, False]``
                - ``2``:
                    - ``kernel_sizes``: ``[3, 3]``
                    - ``out_channels``: ``[128, 128]``
                    - ``strides``: ``[1, 1]``
                    - ``use_se``: ``[False, False]``
                - ``3``:
                    - ``kernel_sizes``: ``[3, 5, 5, 5, 5]``
                    - ``out_channels``: ``[256, 256, 256, 256, 256]``
                    - ``strides``: ``[2, 1, 1, 1, 1]``
                    - ``use_se``: ``[False, False, False, False, False]``
                - ``4``:
                    - ``kernel_sizes``: ``[5, 5, 5, 5]``
                    - ``out_channels``: ``[512, 512, 512, 512]``
                    - ``strides``: ``[1, 1, 1, 1]``
                    - ``use_se``: ``[True, True, False, False]``

    """

    in_channels: int

    @typechecked
    def __init__(
        self,
        scale: float,
        n_branches: int,
        use_detection_backbone: bool,
        max_text_len: int,
        layer_params: list["LayerParamsDict"] | None = None,
        **kwargs,
    ):
        """Initialize the PPLCNetV3 backbone.

        Args:
            scale (float): Scale factor. Defaults to 0.95.
            n_branches (int): Number of convolution branches. Defaults to 4.
            use_detection_backbone (bool): Whether to use the detection backbone. Defaults to False.
            max_text_len (int): Maximum text length. Defaults to 40.
            layer_params (list[LayerParamsDict] | None): Parameters for each LCNetV3 layer.
            **kwargs (Any): Keyword arguments forwarded to the parent class.

        """
        super().__init__(**kwargs)
        layer_params = layer_params or []

        self.scale = scale
        self.use_detection_backbone = use_detection_backbone
        self.n_branches = n_branches

        self.conv = ConvBlock(
            in_channels=self.in_channels,
            out_channels=scale_up(16, self.scale),
            kernel_size=3,
            stride=2,
            padding=1,
            bias=True,
            activation=False,
        )

        blocks: list[LCNetV3Layer] = []
        in_channels = scale_up(16, self.scale)
        for params in layer_params:
            blocks.append(
                LCNetV3Layer(
                    in_channels=in_channels,
                    n_branches=self.n_branches,
                    scale=self.scale,
                    **params,
                )
            )
            in_channels = blocks[-1].out_channels
        self.blocks = nn.ModuleList(blocks)

        if self.use_detection_backbone:
            blocks_out_channels = [
                scale_up(blocks[i].out_channels, self.scale)
                for i in range(1, 5)
            ]

            detecion_out_channels = [
                int(c * self.scale) for c in [16, 24, 56, 480]
            ]

            self.detecion_blocks = nn.ModuleList(
                [
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=True,
                    )
                    for in_channels, out_channels in zip(
                        blocks_out_channels, detecion_out_channels, strict=True
                    )
                ]
            )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, max_text_len))

    def forward(self, x: Tensor) -> list[Tensor]:
        out = []
        x = self.conv(x)
        x = self.blocks[0](x)
        x = self.blocks[1](x)

        out.append(x)
        x = self.blocks[2](x)
        out.append(x)
        x = self.blocks[3](x)
        out.append(x)
        x = self.blocks[4](x)
        out.append(x)

        if self.use_detection_backbone:
            for i in range(4):
                out[i] = self.detecion_blocks[i](out[i])
            return out

        out.append(self.avg_pool(x))

        return out

    @override
    @staticmethod
    def get_variants() -> tuple[str, dict[str, "PPLCNetVariantDict"]]:
        return "rec-light", {
            "rec-light": {
                "scale": 0.95,
                "n_branches": 4,
                "use_detection_backbone": False,
                "layer_params": [
                    {
                        "kernel_sizes": [3],
                        "out_channels": [32],
                        "strides": [1],
                        "use_se": [False],
                    },
                    {
                        "kernel_sizes": [3, 3],
                        "out_channels": [64, 64],
                        "strides": [2, 1],
                        "use_se": [False, False],
                    },
                    {
                        "kernel_sizes": [3, 3],
                        "out_channels": [128, 128],
                        "strides": [1, 1],
                        "use_se": [False, False],
                    },
                    {
                        "kernel_sizes": [3, 5, 5, 5, 5],
                        "out_channels": [256, 256, 256, 256, 256],
                        "strides": [2, 1, 1, 1, 1],
                        "use_se": [False, False, False, False, False],
                    },
                    {
                        "kernel_sizes": [5, 5, 5, 5],
                        "out_channels": [512, 512, 512, 512],
                        "strides": [1, 1, 1, 1],
                        "use_se": [True, True, False, False],
                    },
                ],
            }
        }


class LayerParamsDict(TypedDict):
    kernel_sizes: list[int]
    out_channels: list[int]
    strides: list[int]
    use_se: list[bool]


class PPLCNetVariantDict(TypedDict):
    scale: float
    n_branches: int
    use_detection_backbone: bool
    layer_params: list[LayerParamsDict]
