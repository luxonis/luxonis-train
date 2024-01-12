"""Implementation of the EfficientNet backbone.

Source: U{https://github.com/rwightman/gen-efficientnet-pytorch}
@license: U{Apache 2.0<https://github.com/rwightman/gen-efficientnet-pytorch/blob/master/LICENSE>}
"""

import torch
from torch import Tensor

from .base_node import BaseNode


class EfficientNet(BaseNode[Tensor, list[Tensor]]):
    def __init__(self, download_weights: bool = False, **kwargs):
        """EfficientNet backbone.

        @type download_weights: bool
        @param download_weights: If C{True} download weights from imagenet. Defaults to
            C{False}.
        """
        super().__init__(**kwargs)

        efficientnet_lite0_model = torch.hub.load(
            "rwightman/gen-efficientnet-pytorch",
            "efficientnet_lite0",
            pretrained=download_weights,
        )
        self.out_indices = [1, 2, 4, 6]
        self.backbone = efficientnet_lite0_model

    def forward(self, x: Tensor) -> list[Tensor]:
        outs = []
        x = self.backbone.conv_stem(x)
        x = self.backbone.bn1(x)
        x = self.backbone.act1(x)
        for i, m in enumerate(self.backbone.blocks):
            x = m(x)
            if i in self.out_indices:
                outs.append(x)
        return outs
