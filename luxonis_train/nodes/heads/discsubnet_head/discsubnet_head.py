from typing import Tuple

import torch
from torch import Tensor

from luxonis_train.enums import TaskType
from luxonis_train.nodes.base_node import BaseNode
from luxonis_train.utils import (
    Packet,
)

from .blocks import Decoder, Encoder


class DiscSubNetHead(BaseNode[Tensor, Tensor]):
    in_channels: list[int] | int
    out_channels: list[int] | int
    base_channels: int
    out_features: bool
    tasks: list[TaskType] = [TaskType.SEGMENTATION]

    def __init__(
        self,
        in_channels: list[int] | int = 6,
        out_channels: list[int] | int = 2,
        base_channels: int = 64,
        out_features: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if isinstance(in_channels, list):
            in_channels = in_channels[0] * 2

        base_width = base_channels
        self.encoder_segment = Encoder(in_channels, base_width)
        self.decoder_segment = Decoder(base_width, out_channels=out_channels)
        self.out_features = out_features

    def forward(self, x_tuple: Tuple[Tensor, Tensor]) -> Tensor:
        """Performs the forward pass through the encoder and decoder."""

        if self.export:
            recon, orig = x_tuple
        else:
            recon, orig, an_mask = x_tuple

        x = torch.cat((recon, orig), dim=1)

        b1, b2, b3, b4, b5, b6 = self.encoder_segment(x)
        seg_out = self.decoder_segment(b1, b2, b3, b4, b5, b6)

        if self.export:
            return seg_out
        else:
            return seg_out, recon, orig, an_mask

    def wrap(
        self,
        output: tuple[list[Tensor], list[Tensor], list[Tensor]] | list[Tensor],
    ) -> Packet[Tensor]:
        if self.export:
            return {"segmentation": output}
        else:
            seg_out, orig, recon, an_mask = output
            return {
                "original": orig,
                "reconstructed": recon,
                "segmentation": [seg_out],
                "anomaly_mask": an_mask,
            }
