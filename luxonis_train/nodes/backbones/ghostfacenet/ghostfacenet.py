import math

from luxonis_ml.typing import Kwargs
from torch import Tensor, nn
from typeguard import typechecked
from typing_extensions import override

from luxonis_train.nodes.backbones.micronet.blocks import _make_divisible
from luxonis_train.nodes.base_node import BaseNode
from luxonis_train.nodes.blocks import ConvBlock

from .blocks import GhostBottleneckV2


class GhostFaceNetV2(BaseNode[Tensor, Tensor]):
    """GhostFaceNetsV2 backbone.

    GhostFaceNetsV2 is a convolutional neural network architecture focused on face recognition, but it is
    adaptable to generic embedding tasks. It is based on the GhostNet architecture and uses Ghost BottleneckV2 blocks.

    Source: U{https://github.com/Hazqeel09/ellzaf_ml/blob/main/ellzaf_ml/models/ghostfacenetsv2.py}

    Variants
    ========
    This backbone offers a single variant, V2, which is the default variant.

    @license: U{MIT License
        <https://github.com/Hazqeel09/ellzaf_ml/blob/main/LICENSE>}

    @see: U{GhostFaceNets: Lightweight Face Recognition Model From Cheap Operations
        <https://www.researchgate.net/publication/369930264_GhostFaceNets_Lightweight_Face_Recognition_Model_from_Cheap_Operations>}
    """

    in_channels: int
    in_width: int

    @typechecked
    def __init__(
        self,
        width_multiplier: int,
        kernel_sizes: list[list[int]],
        expand_sizes: list[list[int]],
        output_channels: list[list[int]],
        se_ratios: list[list[float]],
        strides: list[list[int]],
        **kwargs,
    ):
        """
        @type width_multiplier: int
        @param width_multiplier: Width multiplier for the blocks.
        @type kernel_sizes: list[list[int]]
        @param kernel_sizes: List of kernel sizes for block in each stage.
        @type expand_sizes: list[list[int]]
        @param expand_sizes: List of expansion sizes for block in each stage.
        @type output_channels: list[list[int]]
        @param output_channels: List of output channels for block in each stage.
        @type se_ratios: list[list[float]]
        @param se_ratios: List of Squeeze-and-Excitation ratios for block in each stage.
        @type strides: list[list[int]]
        @param strides: List of strides for block in each stage.
        """
        super().__init__(**kwargs)

        output_channel = _make_divisible(int(16 * width_multiplier), 4)
        input_channel = output_channel

        blocks: list[nn.Module] = [
            ConvBlock(
                self.in_channels,
                output_channel,
                kernel_size=3,
                stride=2,
                padding=1,
                activation=nn.PReLU(),
            )
        ]
        layer_id = 0
        for kernel_size, expand_size, output_channel, se_ratio, stride in zip(
            kernel_sizes,
            expand_sizes,
            output_channels,
            se_ratios,
            strides,
            strict=True,
        ):
            layers = []
            for k, ex, oc, sr, st in zip(
                kernel_size,
                expand_size,
                output_channel,
                se_ratio,
                stride,
                strict=True,
            ):
                output_channel = _make_divisible(oc * width_multiplier, 4)
                hidden_channel = _make_divisible(ex * width_multiplier, 4)
                layers.append(
                    GhostBottleneckV2(
                        input_channel,
                        hidden_channel,
                        output_channel,
                        kernel_size=k,
                        stride=st,
                        se_ratio=sr,
                        layer_id=layer_id,
                    )
                )
                input_channel = output_channel
                layer_id += 1
            blocks.append(nn.Sequential(*layers))
        output_channel = _make_divisible(ex * width_multiplier, 4)
        blocks.append(
            ConvBlock(
                input_channel,
                output_channel,
                kernel_size=1,
                activation=nn.PReLU(),
            )
        )

        self.blocks = nn.ModuleList(blocks)

    @override
    def initialize_weights(self, method: str | None = None) -> None:
        super().initialize_weights(method)
        for m in self.modules():
            if isinstance(m, nn.Conv2d | nn.Linear):
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                negative_slope = 0.25
                m.weight.data.normal_(
                    0, math.sqrt(2.0 / (fan_in * (1 + negative_slope**2)))
                )
            if isinstance(m, nn.BatchNorm2d):
                m.momentum = 0.9
                m.eps = 1e-5

    def forward(self, x: Tensor) -> list[Tensor]:
        outputs = []
        for block in self.blocks:
            x = block(x)
            outputs.append(x)
        return outputs

    # TODO: The variant settings for this node are too complicated.
    # Can we simplify them?
    @override
    @staticmethod
    def get_variants() -> tuple[str, dict[str, Kwargs]]:
        return "V2", {
            "V2": {
                "width_multiplier": 1,
                "kernel_sizes": [
                    [3],
                    [3],
                    [3],
                    [5],
                    [5],
                    [3],
                    [3, 3, 3, 3, 3],
                    [5],
                    [5, 5, 5, 5],
                ],
                "expand_sizes": [
                    [16],
                    [48],
                    [72],
                    [72],
                    [120],
                    [240],
                    [200, 184, 184, 480, 672],
                    [672],
                    [960, 960, 960, 960],
                ],
                "output_channels": [
                    [16],
                    [24],
                    [24],
                    [40],
                    [40],
                    [80],
                    [80, 80, 80, 112, 112],
                    [160],
                    [160, 160, 160, 160],
                ],
                "se_ratios": [
                    [0.0],
                    [0.0],
                    [0.0],
                    [0.25],
                    [0.25],
                    [0.0],
                    [0.0, 0.0, 0.0, 0.25, 0.25],
                    [0.25],
                    [0.0, 0.25, 0.0, 0.25],
                ],
                "strides": [
                    [1],
                    [2],
                    [1],
                    [2],
                    [1],
                    [2],
                    [1, 1, 1, 1, 1],
                    [2],
                    [1, 1, 1, 1],
                ],
            }
        }
