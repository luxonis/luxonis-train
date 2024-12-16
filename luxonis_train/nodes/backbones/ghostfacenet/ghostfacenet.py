# Original source: https://github.com/Hazqeel09/ellzaf_ml/blob/main/ellzaf_ml/models/ghostfacenetsv2.py
import math
from typing import Literal

import torch.nn as nn
from torch import Tensor

from luxonis_train.nodes.backbones.ghostfacenet.blocks import (
    GhostBottleneckV2,
    ModifiedGDC,
)
from luxonis_train.nodes.backbones.ghostfacenet.variants import get_variant
from luxonis_train.nodes.backbones.micronet.blocks import _make_divisible
from luxonis_train.nodes.base_node import BaseNode
from luxonis_train.nodes.blocks import ConvModule


class GhostFaceNetsV2(BaseNode[Tensor, list[Tensor]]):
    in_channels: list[int]
    in_width: list[int]

    def __init__(
        self,
        embedding_size=512,
        num_classes=-1,
        variant: Literal["V2"] = "V2",
        *args,
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
        @type num_classes: int
        @param num_classes: Number of classes. Defaults to -1, which leaves the default variant value in. Otherwise it can be used to
            have the network return raw embeddings (=0) or add another linear layer to the network, which is useful for training using
            ArcFace or similar classification-based losses that require the user to drop the last layer of the network.
        @type variant: Literal["V2"]
        @param variant: Variant of the GhostFaceNets embedding model. Defaults to "V2" (which is the only variant available).
        """
        super().__init__(*args, **kwargs)

        image_size = self.in_width[0]
        channels = self.in_channels[0]
        var = get_variant(variant)
        if num_classes >= 0:
            var.num_classes = num_classes
        self.cfgs = var.cfgs

        # Building first layer
        output_channel = _make_divisible(int(16 * var.width), 4)
        self.conv_stem = nn.Conv2d(
            channels, output_channel, 3, 2, 1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.act1 = nn.PReLU()
        input_channel = output_channel

        # Building Ghost BottleneckV2 blocks
        stages = []
        layer_id = 0
        for cfg in self.cfgs:
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
                            args=var.block_args,
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

        # Building pointwise convolution
        pointwise_conv = []
        if var.add_pointwise_conv:
            pointwise_conv.append(
                nn.Conv2d(input_channel, output_channel, 1, 1, 0, bias=True)
            )
            pointwise_conv.append(nn.BatchNorm2d(output_channel))
            pointwise_conv.append(nn.PReLU())
        else:
            pointwise_conv.append(nn.Sequential())

        self.pointwise_conv = nn.Sequential(*pointwise_conv)
        self.classifier = ModifiedGDC(
            image_size,
            output_channel,
            var.num_classes,
            var.dropout,
            embedding_size,
        )

        # Initializing weights
        for m in self.modules():
            if var.init_kaiming:
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    negative_slope = 0.25
                    m.weight.data.normal_(
                        0, math.sqrt(2.0 / (fan_in * (1 + negative_slope**2)))
                    )
            if isinstance(m, nn.BatchNorm2d):
                m.momentum, m.eps = var.bn_momentum, var.bn_epsilon

    def unwrap(self, inputs):
        return [inputs[0]["features"][0]]

    def wrap(self, outputs):
        return {"features": [outputs]}

    def forward(self, inps):
        x = inps[0]
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.blocks(x)
        x = self.pointwise_conv(x)
        x = self.classifier(x)
        return x
