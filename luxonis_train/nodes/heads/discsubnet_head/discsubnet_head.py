from typing import Tuple

import torch
from torch import Tensor

from luxonis_train.enums import TaskType
from luxonis_train.nodes.base_node import BaseNode
from luxonis_train.utils import (
    Packet,
)

from .blocks import Decoder, Encoder, NanoDecoder, NanoEncoder


class DiscSubNetHead(BaseNode[Tensor, Tensor]):
    in_channels: list[int] | int
    out_channels: list[int] | int
    base_channels: int
    tasks: list[TaskType] = [TaskType.SEGMENTATION]

    def __init__(
        self,
        in_channels: list[int] | int = 6,
        out_channels: list[int] | int = 2,
        base_channels: int = 64,
        variant: str = "L",
        **kwargs,
    ):
        """
        DiscSubNetHead: A discriminative sub-network that detects and segments anomalies in images.

        This model is designed to take an input image and generate a mask that highlights anomalies or
        regions of interest based on reconstruction. The encoder extracts relevant features from the
        input, while the decoder generates a mask that identifies areas of anomalies by distinguishing
        between the reconstructed image and the input.

        The network has two variants:
        - "L" (large): uses a full encoder-decoder architecture with more filters.
        - "N" (nano): a lightweight version with fewer filters for more efficient processing.

        @type in_channels: list[int] | int
        @param in_channels: Number of input channels for the encoder. Defaults to 6.

        @type out_channels: list[int] | int
        @param out_channels: Number of output channels for the decoder. Defaults to 2 (for segmentation masks).

        @type base_channels: int
        @param base_channels: The base number of filters used in the encoder and decoder blocks. Determines model size.

        @type variant: str
        @param variant: The variant of the DiscSubNetHead to use. "L" for large, "N" for nano (lightweight). Defaults to "L".

        @type kwargs: Any
        @param kwargs: Additional arguments to be passed to the BaseNode class.
        """
        super().__init__(**kwargs)

        if isinstance(in_channels, list):
            in_channels = in_channels[0] * 2

        base_width = base_channels
        if variant == "L":
            self.encoder_segment = Encoder(in_channels, base_width)
            self.decoder_segment = Decoder(base_width, out_channels)
        elif variant == "N":
            nano_base_width = base_width // 2
            self.encoder_segment = NanoEncoder(in_channels, nano_base_width)
            self.decoder_segment = NanoDecoder(nano_base_width, out_channels)

    def forward(self, x_tuple: Tuple[Tensor, Tensor]) -> Tensor:
        """Performs the forward pass through the encoder and decoder."""

        recon, input = x_tuple

        x = torch.cat((recon, input), dim=1)

        seg_out = self.decoder_segment(*self.encoder_segment(x))

        if self.export:
            return seg_out
        else:
            return seg_out, recon

    def wrap(
        self,
        output: tuple[list[Tensor], list[Tensor]] | list[Tensor],
    ) -> Packet[Tensor]:
        if self.export:
            return {"segmentation": output}
        else:
            seg_out, recon = output
            return {
                "reconstructed": [recon],
                "segmentation": [seg_out],
            }
