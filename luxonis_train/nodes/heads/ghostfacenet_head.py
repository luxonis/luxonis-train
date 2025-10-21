# Original source: https://github.com/Hazqeel09/ellzaf_ml/blob/main/ellzaf_ml/models/ghostfacenetsv2.py
import math

from torch import Tensor, nn
from typing_extensions import override

from luxonis_train.nodes.blocks.blocks import ConvBlock
from luxonis_train.nodes.heads.base_head import BaseHead
from luxonis_train.tasks import Tasks


class GhostFaceNetHead(BaseHead):
    in_channels: int
    in_width: int
    task = Tasks.EMBEDDINGS

    def __init__(
        self,
        embedding_size: int = 512,
        cross_batch_memory_size: int | None = None,
        dropout: float = 0.2,
        **kwargs,
    ):
        """GhostFaceNet backbone.

        GhostFaceNet is a convolutional neural network architecture focused on face recognition, but it is
        adaptable to generic embedding tasks. It is based on the GhostNet architecture and uses Ghost BottleneckV2 blocks.

        Source: U{https://github.com/Hazqeel09/ellzaf_ml/blob/main/ellzaf_ml/models/ghostfacenetsv2.py}

        @license: U{MIT License
            <https://github.com/Hazqeel09/ellzaf_ml/blob/main/LICENSE>}

        @see: U{GhostFaceNets: Lightweight Face Recognition Model From Cheap Operations
            <https://www.researchgate.net/publication/369930264_GhostFaceNets_Lightweight_Face_Recognition_Model_from_Cheap_Operations>}

        @type embedding_size: int
        @param embedding_size: Size of the embedding. Defaults to 512.
        @type cross_batch_memory_size: int | None
        @param cross_batch_memory_size: Size of the cross-batch memory. Defaults to None.
        @type dropout: float
        @param dropout: Dropout rate. Defaults to 0.2.
        """
        super().__init__(**kwargs)
        self.embedding_size = embedding_size
        self.cross_batch_memory_size = cross_batch_memory_size
        _, H, W = self.original_in_shape

        self.head = nn.Sequential(
            ConvBlock(
                self.in_channels,
                self.in_channels,
                kernel_size=(
                    H // 32 if H % 32 == 0 else H // 32 + 1,
                    W // 32 if W % 32 == 0 else W // 32 + 1,
                ),
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

    def forward(self, x: Tensor) -> Tensor:
        return self.head(x)

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
