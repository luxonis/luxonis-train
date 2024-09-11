from typing import Any

import torch
from torch import Tensor, nn

from luxonis_train.nodes.base_node import BaseNode


class EfficientNet(BaseNode[Tensor, list[Tensor]]):
    attach_index: int = -1

    def __init__(
        self,
        download_weights: bool = False,
        out_indices: list[int] | None = None,
        **kwargs: Any,
    ):
        """EfficientNet backbone.

        EfficientNet is a convolutional neural network architecture and scaling method that uniformly scales all dimensions of depth/width/resolution using a compound coefficient. Unlike conventional practice that arbitrary scales these factors, the EfficientNet scaling method uniformly scales network width, depth, and resolution with a set of fixed scaling coefficients.

        Source: U{https://github.com/rwightman/gen-efficientnet-pytorch}

        @license: U{Apache License, Version 2.0
            <https://github.com/rwightman/gen-efficientnet-pytorch/blob/master/LICENSE>}

        @see: U{https://paperswithcode.com/method/efficientnet}
        @see: U{EfficientNet: Rethinking Model Scaling for
            Convolutional Neural Networks
            <https://arxiv.org/abs/1905.11946>}
        @type download_weights: bool
        @param download_weights: If C{True} download weights from imagenet. Defaults to
            C{False}.
        @type out_indices: list[int] | None
        @param out_indices: Indices of the output layers. Defaults to [0, 1, 2, 4, 6].
        """
        super().__init__(**kwargs)

        self.backbone: nn.Module = torch.hub.load(  # type: ignore
            "rwightman/gen-efficientnet-pytorch",
            "efficientnet_lite0",
            pretrained=download_weights,
        )
        self.out_indices = out_indices or [0, 1, 2, 4, 6]

    def forward(self, inputs: Tensor) -> list[Tensor]:
        x = self.backbone.conv_stem(inputs)
        x = self.backbone.bn1(x)
        x = self.backbone.act1(x)

        outs: list[Tensor] = []

        for i, layer in enumerate(self.backbone.blocks):
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)

        return outs
