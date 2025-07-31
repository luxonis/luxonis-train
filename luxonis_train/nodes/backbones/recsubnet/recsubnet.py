from typing import Literal, TypeAlias

from torch import Tensor

from luxonis_train.nodes.base_node import BaseNode

from .blocks import Decoder, Encoder, NanoDecoder, NanoEncoder

VariantLiteral: TypeAlias = Literal["n", "l"]


def get_variant(variant: VariantLiteral) -> int:
    """Returns the base width for the specified variant."""
    variants = {
        "n": 64,
        "l": 128,
    }

    if variant not in variants:
        raise ValueError(
            f"Variant should be one of {list(variants.keys())}, got '{variant}'."
        )

    return variants[variant]


class RecSubNet(BaseNode[Tensor, tuple[Tensor, Tensor]]):
    in_channels: int
    out_channels: int
    base_width: int

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_width: int | None = None,
        variant: VariantLiteral = "l",
        **kwargs,
    ):
        """RecSubNet: A reconstruction sub-network that consists of an
        encoder and a decoder.

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

        @type variant: Literal["n", "l"]
        @param variant: The variant of the RecSubNet to use. "l" for large, "n" for nano (lightweight). Defaults to "l".
        """
        super().__init__(**kwargs)

        self.base_width = (
            base_width if base_width is not None else get_variant(variant)
        )

        if variant == "l":
            self.encoder = Encoder(in_channels, self.base_width)
            self.decoder = Decoder(self.base_width, out_channels=out_channels)
        elif variant == "n":
            self.encoder = NanoEncoder(in_channels, self.base_width)
            self.decoder = NanoDecoder(
                self.base_width, out_channels=out_channels
            )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Performs the forward pass through the encoder and decoder."""
        b5 = self.encoder(x)
        output = self.decoder(b5)

        return output, x
