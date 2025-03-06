from luxonis_ml.typing import Kwargs
from torch import Tensor, nn
from typeguard import typechecked
from typing_extensions import override

from luxonis_train.nodes.base_node import BaseNode
from luxonis_train.nodes.blocks import ConvModule

from .blocks import LCNetV3Block


class PPLCNetV3(BaseNode[Tensor, list[Tensor]]):
    """PPLCNetV3 backbone.

    Variants
    --------
    Only one variant is available, "rec-light".

    @see: U{Adapted from <https://github.com/PaddlePaddle/PaddleOCR/
        blob/main/ppocr/modeling/backbones/rec_lcnetv3.py>}
    @see: U{Original code
        <https://github.com/PaddlePaddle/PaddleOCR>}
    @license: U{Apache License, Version 2.0
        <https://github.com/PaddlePaddle/PaddleOCR/blob/main/LICENSE
        >}
    """

    default_variant = "rec-light"
    in_channels: int

    @typechecked
    def __init__(
        self,
        scale: float,
        n_branches: int,
        use_detection_backbone: bool,
        kernel_sizes: list[list[int]],
        in_channels: list[list[int]],
        out_channels: list[list[int]],
        strides: list[list[int]],
        use_se: list[list[bool]],
        max_text_len: int,
        **kwargs,
    ):
        """
        @type scale: float
        @param scale: Scale factor. Defaults to 0.95.
        @type n_branches: int
        @param n_branches: Number of convolution branches.
            Defaults to 4.
        @type use_detection_backbone: bool
        @param use_detection_backbone: Whether to use the detection backbone.
            Defaults to False.
        @type max_text_len: int
        @param max_text_len: Maximum text length. Defaults to 40.
        """
        super().__init__(**kwargs)

        self.scale = scale
        self.use_detection_backbone = use_detection_backbone
        self.n_branches = n_branches

        self.conv = ConvModule(
            in_channels=self.in_channels,
            out_channels=_make_divisible(16 * self.scale),
            kernel_size=3,
            stride=2,
            padding=1,
            bias=True,
            activation=None,
        )
        blocks = []
        for (
            in_channel,
            out_channel,
            kernel_size,
            stride,
            _use_se,
        ) in zip(
            in_channels,
            out_channels,
            kernel_sizes,
            strides,
            use_se,
            strict=True,
        ):
            layer = []
            for ic, oc, ks, s, se in zip(
                in_channel,
                out_channel,
                kernel_size,
                stride,
                _use_se,
                strict=True,
            ):
                layer.append(
                    LCNetV3Block(
                        in_channels=_make_divisible(ic * self.scale),
                        out_channels=_make_divisible(oc * self.scale),
                        kernel_size=ks,
                        stride=s,
                        use_se=se,
                        n_branches=self.n_branches,
                    )
                )
            blocks.append(nn.Sequential(*layer))
        self.blocks = nn.ModuleList(blocks)

        if self.use_detection_backbone:
            blocks_out_channels = [
                _make_divisible(out_channels[i][-1] * self.scale)
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
    def get_variants() -> dict[str, Kwargs]:
        return {
            "rec-light": {
                "scale": 0.95,
                "n_branches": 4,
                "use_detection_backbone": False,
                "kernel_sizes": [
                    [3],
                    [3, 3],
                    [3, 3],
                    [3, 5, 5, 5, 5],
                    [5, 5, 5, 5],
                ],
                "in_channels": [
                    [16],
                    [32, 64],
                    [64, 128],
                    [128, 256, 256, 256, 256],
                    [256, 512, 512, 512],
                ],
                "out_channels": [
                    [32],
                    [64, 64],
                    [128, 128],
                    [256, 256, 256, 256, 256],
                    [512, 512, 512, 512],
                ],
                "strides": [
                    [1],
                    [2, 1],
                    [1, 1],
                    [2, 1, 1, 1, 1],
                    [1, 1, 1, 1],
                ],
                "use_se": [
                    [False],
                    [False, False],
                    [False, False],
                    [False, False, False, False, False],
                    [True, True, False, False],
                ],
            }
        }


def _make_divisible(
    v: float, divisor: int = 16, min_value: int | None = None
) -> int:
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v
