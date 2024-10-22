from typing import Tuple

from torch import Tensor

from luxonis_train.nodes.base_node import BaseNode

from .blocks import Decoder, Encoder, NanoDecoder, NanoEncoder


class RecSubNet(BaseNode[Tensor, Tuple[Tensor, Tensor, Tensor]]):
    in_channels: int
    out_channels: int
    base_width: int

    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        base_width=128,
        variant="L",
        **kwargs,
    ):
        """
        RecSubNet: A reconstruction sub-network that consists of an encoder and a decoder.

        This model is designed to reconstruct the original image from an input image that contains noise or anomalies.
        The encoder extracts relevant features from the noisy input, and the decoder attempts to reconstruct the clean
        version of the image by eliminating the noise or anomalies.

        This architecture is based on the paper:
        "Data-Efficient Image Transformers: A Deeper Look" (https://arxiv.org/abs/2108.07610).

        @type in_channels: int
        @param in_channels: Number of input channels for the encoder. Defaults to 3.

        @type out_channels: int
        @param out_channels: Number of output channels for the decoder. Defaults to 3.

        @type base_width: int
        @param base_width: The base width of the network. Determines the number of filters in the encoder and decoder.

        @type variant: str
        @param variant: The variant of the RecSubNet to use. Defaults to "L".

        @type kwargs: Any
        @param kwargs: Additional arguments to be passed to the BaseNode class.
        """
        super().__init__(**kwargs)

        if variant == "L":
            self.encoder = Encoder(in_channels, base_width)
            self.decoder = Decoder(base_width, out_channels=out_channels)
        elif variant == "N":
            self.encoder = NanoEncoder(in_channels, base_width // 2)
            self.decoder = NanoDecoder(
                base_width // 2, out_channels=out_channels
            )

    def forward(self, x: Tensor) -> Tensor:
        """Performs the forward pass through the encoder and decoder."""
        b5 = self.encoder(x)
        output = self.decoder(b5)

        return output, x
