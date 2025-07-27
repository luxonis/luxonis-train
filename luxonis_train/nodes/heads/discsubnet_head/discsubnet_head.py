import torch
from luxonis_ml.typing import Kwargs
from torch import Tensor
from typing_extensions import override

from luxonis_train.nodes.blocks import UNetDecoder, UNetEncoder
from luxonis_train.nodes.heads import BaseHead
from luxonis_train.tasks import Tasks
from luxonis_train.typing import Packet


class DiscSubNetHead(BaseHead[Tensor, Tensor]):
    task = Tasks.ANOMALY_DETECTION

    in_channels: list[int] | int
    base_channels: int

    def __init__(
        self,
        base_channels: int,
        width_multipliers: list[float],
        in_channels: list[int] | int = 6,
        out_channels: int = 2,
        **kwargs,
    ):
        """
        DiscSubNetHead: A discriminative sub-network that detects and segments anomalies in images.

        This model is designed to take an input image and generate a mask that highlights anomalies or
        regions of interest based on reconstruction. The encoder extracts relevant features from the
        input, while the decoder generates a mask that identifies areas of anomalies by distinguishing
        between the reconstructed image and the input.

        @type in_channels: list[int] | int
        @param in_channels: Number of input channels for the encoder. Defaults to 6.

        @type out_channels: int
        @param out_channels: Number of output channels for the decoder.
            Defaults to 2 (for segmentation masks).

        @type base_channels: int
        @param base_channels: The base number of filters used in the encoder and decoder blocks. If None, it is determined based on the variant.
        """
        super().__init__(**kwargs)

        if isinstance(in_channels, list):
            in_channels = in_channels[0] * 2

        self.encoder_segment = UNetEncoder(
            in_channels, base_channels, width_multipliers
        )
        self.decoder_segment = UNetDecoder(
            base_channels, out_channels, width_multipliers
        )

    def forward(self, inputs: list[Tensor]) -> tuple[Tensor, Tensor]:
        """Performs the forward pass through the encoder and decoder."""

        reconstruction, x = inputs
        x = torch.cat([reconstruction, x], dim=1)
        seg_out = self.decoder_segment(self.encoder_segment(x))

        return seg_out, reconstruction

    @override
    def wrap(self, output: tuple[Tensor, Tensor]) -> Packet[Tensor]:
        """Wraps the output into a packet."""
        seg_out, recon = output
        if self.export:
            return {"segmentation": seg_out}
        seg_out, recon = output
        return {"reconstructed": recon, "segmentation": seg_out}

    @override
    def get_custom_head_config(self) -> dict:
        """Returns custom head configuration.

        @rtype: dict
        @return: Custom head configuration.
        """
        return {}

    @override
    @staticmethod
    def get_variants() -> tuple[str, dict[str, Kwargs]]:
        return "n", {
            "n": {
                "base_channels": 32,
                "width_multipliers": [1, 1.1],
            },
            "l": {
                "base_channels": 64,
                "width_multipliers": [1, 2, 4, 8, 8],
            },
        }
