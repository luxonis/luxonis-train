# Original source: https://github.com/Hazqeel09/ellzaf_ml/blob/main/ellzaf_ml/models/ghostfacenetsv2.py
import math
from typing import Literal

import torch.nn as nn
from torch import Tensor

from luxonis_train.enums import Metadata
from luxonis_train.nodes.backbones.ghostfacenet.variants import get_variant
from luxonis_train.nodes.backbones.micronet.blocks import _make_divisible
from luxonis_train.nodes.base_node import BaseNode
from luxonis_train.nodes.blocks import ConvModule

from .blocks import GhostBottleneckV2, ModifiedGDC


class GhostFaceNetV2(BaseNode[Tensor, Tensor]):
    in_channels: int
    in_width: int
    tasks = [Metadata("id")]

    def __init__(
        self,
        embedding_size: int = 512,
        variant: Literal["V2"] = "V2",
        **kwargs,
    ):
        """GhostFaceNetsV2 backbone.

        GhostFaceNetsV2 is a convolutional neural network architecture focused on face recognition, but it is
        adaptable to generic embedding tasks. It is based on the GhostNet architecture and uses Ghost BottleneckV2 blocks.

        Source: U{https://github.com/Hazqeel09/ellzaf_ml/blob/main/ellzaf_ml/models/ghostfacenetsv2.py}

        @license: U{MIT License
            <https://github.com/Hazqeel09/ellzaf_ml/blob/main/LICENSE>}

        @see: U{GhostFaceNets: Lightweight Face Recognition Model From Cheap Operations
            <https://www.researchgate.net/publication/369930264_GhostFaceNets_Lightweight_Face_Recognition_Model_from_Cheap_Operations>}

        @type embedding_size: int
        @param embedding_size: Size of the embedding. Defaults to 512.
        @type variant: Literal["V2"]
        @param variant: Variant of the GhostFaceNets embedding model. Defaults to "V2" (which is the only variant available).
        """
        super().__init__(**kwargs)
        self.embedding_size = embedding_size

        image_size = self.in_width
        var = get_variant(variant)
        output_channel = _make_divisible(int(16 * var.width), 4)
        input_channel = output_channel

        stages: list[nn.Module] = [
            ConvModule(
                self.in_channels,
                output_channel,
                kernel_size=3,
                stride=2,
                padding=1,
                activation=nn.PReLU(),
            )
        ]
        layer_id = 0
        for cfg in var.block_configs:
            layers = []
            for b_cfg in cfg:
                output_channel = _make_divisible(
                    b_cfg.output_channels * var.width, 4
                )
                hidden_channel = _make_divisible(
                    b_cfg.expand_size * var.width, 4
                )
                if var.block == GhostBottleneckV2:
                    layers.append(
                        var.block(
                            input_channel,
                            hidden_channel,
                            output_channel,
                            b_cfg.kernel_size,
                            b_cfg.stride,
                            se_ratio=b_cfg.se_ratio,
                            layer_id=layer_id,
                        )
                    )
                input_channel = output_channel
                layer_id += 1
            stages.append(nn.Sequential(*layers))

        output_channel = _make_divisible(b_cfg.expand_size * var.width, 4)
        stages.append(
            nn.Sequential(
                ConvModule(
                    input_channel,
                    output_channel,
                    kernel_size=1,
                    activation=nn.PReLU(),
                )
            )
        )

        self.blocks = nn.Sequential(*stages)

        self.head = ModifiedGDC(
            image_size,
            output_channel,
            var.dropout,
            embedding_size,
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                negative_slope = 0.25
                m.weight.data.normal_(
                    0, math.sqrt(2.0 / (fan_in * (1 + negative_slope**2)))
                )
            if isinstance(m, nn.BatchNorm2d):
                m.momentum = 0.9
                m.eps = 1e-5

    def forward(self, x: Tensor) -> Tensor:
        x = self.blocks(x)
        x = self.head(x)
        return x
