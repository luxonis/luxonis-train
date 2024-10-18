from typing import Tuple

import torch
from torch import Tensor

from luxonis_train.nodes.base_node import BaseNode

from .blocks import Decoder, Encoder
from .utils import apply_anomaly_to_batch


class RecSubNet(BaseNode[Tensor, Tuple[Tensor, Tensor, Tensor]]):
    in_channels: int
    out_channels: int
    base_width: int

    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        base_width=128,
        anomaly_dataset_dir=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.encoder = Encoder(in_channels, base_width)
        self.decoder = Decoder(base_width, out_channels=out_channels)
        self.anomaly_dataset_dir = anomaly_dataset_dir

    def forward(self, x: Tensor) -> Tensor:
        """Performs the forward pass through the encoder and decoder."""
        if not self.export:
            if torch.rand(1).item() < 0.5:
                x, an_mask = apply_anomaly_to_batch(
                    x, self.anomaly_dataset_dir, x.device
                )
            else:
                h, w = x.shape[-2:]
                an_mask = torch.zeros((x.shape[0], h, w), device=x.device)
            an_mask = an_mask.unsqueeze(1).long()

        # if self.training:
        #     import matplotlib.pyplot as plt

        #     # mask is (batch, h, w)
        #     mask = an_mask[0][0].cpu().numpy()
        #     plt.imshow(mask)
        #     plt.show(block=True)

        #     # imagenet denorm on x[0]
        #     x_plt = x[0].cpu().numpy()
        #     x_plt = x_plt.transpose(1, 2, 0)
        #     x_plt = x_plt * 0.229 + 0.485
        #     plt.imshow(x_plt)
        #     plt.show(block=True)

        b5 = self.encoder(x)

        output = self.decoder(b5)
        if self.export:
            return output, x
        else:
            return output, x, an_mask
