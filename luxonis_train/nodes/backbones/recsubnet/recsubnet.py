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
    """RecSubNet reconstruction backbone.

    RecSubNet is an encoder-decoder node for anomaly reconstruction flows
    that outputs both the reconstruction and original input tensor.

    Metadata:
        - Node type: backbone
        - Registry name: ``RecSubNet``
        - Task: None
        - Attach index: ``-1``
        - Inputs: ``features`` tensor
        - Outputs: ``reconstruction`` tensor and ``original`` tensor

    Provenance:
        - Source: Internal
        - License: Project license
        - Implementation notes: Local reconstruction-specific backbone using
          ``SimpleEncoder`` and ``SimpleDecoder``.

    Variants:
        - ``"n"``:
            - Default: no
            - Aliases: None
            - Parameters:
                - ``base_channels``: ``64``
                - ``width_multipliers``: ``[1, 1.1]``
        - ``"l"``:
            - Default: yes
            - Aliases: None
            - Parameters:
                - ``base_channels``: ``128``
                - ``width_multipliers``: ``[1, 2, 4, 8]``

    """

    in_channels: int

    def __init__(
        self,
        base_channels: int = 128,
        width_multipliers: list[float] | None = None,
        out_channels: int = 3,
        **kwargs,
    ):
        """RecSubNet reconstruction backbone with an encoder and
        decoder.

        This model is designed to reconstruct the original image from an input image that contains noise or anomalies.
        The encoder extracts relevant features from the noisy input, and the decoder attempts to reconstruct the clean
        version of the image by eliminating the noise or anomalies.

        This architecture is based on the paper:
        "Data-Efficient Image Transformers: A Deeper Look" (https://arxiv.org/abs/2108.07610).

        Args:
            base_channels (int): The base width of the network. Determines the number of filters in the encoder and decoder.
            width_multipliers (list[float] | None): Width multipliers for encoder and decoder stages. Defaults to [1, 2, 4, 8].
            out_channels (int): Number of output channels for the decoder. Defaults to 3.
            **kwargs (Any): Keyword arguments forwarded to the parent class.

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
        """Perform the forward pass through the encoder and decoder."""
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
