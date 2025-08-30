from luxonis_ml.typing import Kwargs
from torch import Tensor
from typing_extensions import override

from luxonis_train.nodes.base_node import BaseNode
from luxonis_train.nodes.blocks import SimpleDecoder, SimpleEncoder
from luxonis_train.typing import Packet


# NOTE: This is not really a backbone in the traditional sense.
# It does not output feature maps for further processing by
# an arbitrary head. This node is intended to be used specifically
# with the DiscSubNetHead for anomaly detection tasks.
class RecSubNet(BaseNode):
    in_channels: int

    def __init__(
        self,
        base_channels: int = 128,
        width_multipliers: list[float] | None = None,
        out_channels: int = 3,
        **kwargs,
    ):
        """RecSubNet: A reconstruction sub-network that consists of an
        encoder and a decoder.

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
        width_multipliers = width_multipliers or [1, 2, 4, 8]

        self.encoder = SimpleEncoder(
            self.in_channels,
            base_channels,
            width_multipliers,
            n_convolutions=2,
        )
        self.decoder = SimpleDecoder(
            base_channels,
            out_channels=out_channels,
            encoder_width_multipliers=width_multipliers,
        )

    def forward(self, x: Tensor) -> Packet[Tensor]:
        """Performs the forward pass through the encoder and decoder."""
        return {
            "reconstruction": self.decoder(self.encoder(x)),
            "original": x,
        }

    @staticmethod
    @override
    def get_variants() -> tuple[str, dict[str, Kwargs]]:
        return "l", {
            "n": {
                "base_channels": 64,
                "width_multipliers": [1, 1.1],
            },
            "l": {
                "base_channels": 128,
                "width_multipliers": [1, 2, 4, 8],
            },
        }
