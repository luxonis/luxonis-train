from luxonis_ml.typing import Kwargs
from torch import Tensor, nn
from typing_extensions import override

from luxonis_train.nodes.base_node import BaseNode

from .blocks import Decoder, Encoder, NanoDecoder, NanoEncoder


class RecSubNet(BaseNode[Tensor, tuple[Tensor, Tensor]]):
    default_variant = "n"

    in_channels: int

    def __init__(
        self,
        base_channels: int = 64,
        out_channels: int = 3,
        encoder: type[nn.Module] = Encoder,
        decoder: type[nn.Module] = Decoder,
        **kwargs,
    ):
        """
        RecSubNet: A reconstruction sub-network that consists of an encoder and a decoder.

        This model is designed to reconstruct the original image from an input image that contains noise or anomalies.
        The encoder extracts relevant features from the noisy input, and the decoder attempts to reconstruct the clean
        version of the image by eliminating the noise or anomalies.

        This architecture is based on the paper:
        "Data-Efficient Image Transformers: A Deeper Look" (https://arxiv.org/abs/2108.07610).

        @type out_channels: int
        @param out_channels: Number of output channels for the decoder. Defaults to 3.

        @type base_channels: int
        @param base_channels: The base width of the network.
            Determines the number of filters in the encoder and decoder.

        @type encoder: nn.Module
        @param encoder: The encoder block to use. Defaults to Encoder.

        @type decoder: nn.Module
        @param decoder: The decoder block to use. Defaults to Decoder.
        """
        super().__init__(**kwargs)

        self.encoder = encoder(self.in_channels, base_channels)
        self.decoder = decoder(base_channels, out_channels=out_channels)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Performs the forward pass through the encoder and decoder."""
        b5 = self.encoder(x)
        output = self.decoder(b5)

        return output, x

    @override
    @staticmethod
    def get_variants() -> dict[str, Kwargs]:
        return {
            "n": {
                "base_channels": 64,
                "encoder": NanoEncoder,
                "decoder": NanoDecoder,
            },
            "l": {
                "base_channels": 128,
                "encoder": Encoder,
                "decoder": Decoder,
            },
        }
