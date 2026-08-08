from typing import Literal

import torchvision
from torch import Tensor

from luxonis_train.nodes.base_node import BaseNode


class MobileNetV2(BaseNode):
    """MobileNetV2 backbone."""

    def __init__(
        self,
        out_indices: list[int] | None = None,
        weights: Literal["download", "none"] | None = None,
        **kwargs,
    ):
        """MobileNetV2 backbone.

        This class implements the MobileNetV2 model as described in:
        `MobileNetV2: Inverted Residuals and Linear Bottlenecks
        <https://arxiv.org/pdf/1801.04381v4>`_ by Sandler I{et al.}

        The network consists of an initial fully convolutional layer,
        followed by 19 bottleneck residual blocks, and a final 1x1
        convolution. It can be used as a feature extractor for tasks
        like image classification, object detection, and semantic
        segmentation.

        Key features:
            - Inverted residual structure with linear bottlenecks
            - Depth-wise separable convolutions for efficiency
            - Configurable width multiplier and input resolution

        Args:
            out_indices: Indices of the output layers. Defaults to
                [3, 6, 13, 18].
            weights: Whether to load pretrained weights.
            **kwargs: Base node arguments.

        """
        super().__init__(**kwargs)

        self.backbone = torchvision.models.mobilenet_v2(
            weights="DEFAULT" if weights == "download" else None
        )
        self.out_indices = out_indices or [3, 6, 13, 18]

    def forward(self, inputs: Tensor) -> list[Tensor]:
        r"""Extract a multi-scale MobileNetV2 feature pyramid.

        Args:
            inputs: Image batch to encode.

        Returns:
            Feature maps from the configured output stages, ordered by
            increasing depth.

        .. shape-contract::

            Inputs
                :math:`\mathrm{inputs}`
                    :math:`(B, C_{\mathrm{in}}, H_{\mathrm{in}}, W_{\mathrm{in}})`

            Outputs
                :math:`\mathrm{features}_{i}` (:math:`i = 0, \ldots, N - 1`)
                    :math:`(B, C_{i}, H_{i}, W_{i})`

            Symbols
                :math:`N`
                    Number of tensors in the feature sequence.

        """  # noqa: E501
        outs: list[Tensor] = []
        for i, layer in enumerate(self.backbone.features):
            inputs = layer(inputs)
            if i in self.out_indices:
                outs.append(inputs)

        return outs
