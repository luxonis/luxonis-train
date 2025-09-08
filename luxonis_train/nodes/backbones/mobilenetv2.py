from typing import Literal

import torchvision
from torch import Tensor

from luxonis_train.nodes.base_node import BaseNode


class MobileNetV2(BaseNode):
    def __init__(
        self,
        out_indices: list[int] | None = None,
        weights: Literal["download", "none"] | None = None,
        **kwargs,
    ):
        """MobileNetV2 backbone.

        This class implements the MobileNetV2 model as described in:
        U{MobileNetV2: Inverted Residuals and Linear Bottlenecks <https://arxiv.org/pdf/1801.04381v4>} by Sandler I{et al.}

        The network consists of an initial fully convolutional layer, followed by
        19 bottleneck residual blocks, and a final 1x1 convolution. It can be used
        as a feature extractor for tasks like image classification, object detection,
        and semantic segmentation.

        Key features:
            - Inverted residual structure with linear bottlenecks
            - Depth-wise separable convolutions for efficiency
            - Configurable width multiplier and input resolution

        @type out_indices: list[int] | None
        @param out_indices: Indices of the output layers. Defaults to [3, 6, 13, 18].
        """
        super().__init__(**kwargs)

        self.backbone = torchvision.models.mobilenet_v2(
            weights="DEFAULT" if weights == "download" else None
        )
        self.out_indices = out_indices or [3, 6, 13, 18]

    def forward(self, inputs: Tensor) -> list[Tensor]:
        outs: list[Tensor] = []
        for i, layer in enumerate(self.backbone.features):
            inputs = layer(inputs)
            if i in self.out_indices:
                outs.append(inputs)

        return outs
