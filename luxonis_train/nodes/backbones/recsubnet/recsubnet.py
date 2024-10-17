from typing import Tuple

import torch
from torch import Tensor

from luxonis_train.nodes.base_node import BaseNode

from .blocks import Decoder, Encoder


class RecSubNet(BaseNode[Tensor, Tuple[Tensor, Tensor, Tensor]]):
    in_channels: int
    out_channels: int
    base_width: int

    def __init__(
        self, in_channels=3, out_channels=3, base_width=128, **kwargs
    ):
        super().__init__(**kwargs)

        self.encoder = Encoder(in_channels, base_width)
        self.decoder = Decoder(base_width, out_channels=out_channels)

    def forward(self, x: Tensor) -> Tensor:
        """Performs the forward pass through the encoder and decoder."""
        b5 = self.encoder(x)

        #### dummy mask
        h, w = x.shape[-2:]
        # 2chanel zero mask with h,w
        an_mask = torch.zeros(
            (x.shape[0], h, w), device=x.device
        )  # (bs, h, w)!!
        # no more dummy mask

        output = self.decoder(b5)
        if self.export:
            return output, x
        else:
            return output, x, an_mask
