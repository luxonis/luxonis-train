# Original source: https://github.com/Hazqeel09/ellzaf_ml/blob/main/ellzaf_ml/models/ghostfacenetsv2.py
import math

import torch.nn as nn
from torch import Tensor

from luxonis_train.enums import Metadata
from luxonis_train.nodes.base_node import BaseNode
from luxonis_train.nodes.blocks.blocks import ConvModule


class GhostFaceNetHead(BaseNode[Tensor, list[Tensor]]):
    in_channels: int
    in_width: int
    tasks = [Metadata("id")]

    def __init__(
        self, embedding_size: int = 512, dropout: float = 0.2, **kwargs
    ):
        """GhostFaceNetV2 backbone.

        GhostFaceNetV2 is a convolutional neural network architecture focused on face recognition, but it is
        adaptable to generic embedding tasks. It is based on the GhostNet architecture and uses Ghost BottleneckV2 blocks.

        Source: U{https://github.com/Hazqeel09/ellzaf_ml/blob/main/ellzaf_ml/models/ghostfacenetsv2.py}

        @license: U{MIT License
            <https://github.com/Hazqeel09/ellzaf_ml/blob/main/LICENSE>}

        @see: U{GhostFaceNets: Lightweight Face Recognition Model From Cheap Operations
            <https://www.researchgate.net/publication/369930264_GhostFaceNets_Lightweight_Face_Recognition_Model_from_Cheap_Operations>}

        @type embedding_size: int
        @param embedding_size: Size of the embedding. Defaults to 512.
        """
        super().__init__(**kwargs)
        self.embedding_size = embedding_size
        image_size = self.original_in_shape[1]

        self.head = nn.Sequential(
            ConvModule(
                self.in_channels,
                self.in_channels,
                kernel_size=(image_size // 32)
                if image_size % 32 == 0
                else (image_size // 32 + 1),
                groups=self.in_channels,
                activation=False,
            ),
            nn.Dropout(dropout),
            nn.Conv2d(
                self.in_channels, embedding_size, kernel_size=1, bias=False
            ),
            nn.Flatten(),
            nn.BatchNorm1d(embedding_size),
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
        return self.head(x)
