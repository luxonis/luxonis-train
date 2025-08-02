from typing import Literal, TypeAlias

import torch
from torch import Tensor

from luxonis_train.nodes.heads import BaseHead
from luxonis_train.tasks import Tasks
from luxonis_train.typing import Packet

from .blocks import Decoder, Encoder, NanoDecoder, NanoEncoder

VariantLiteral: TypeAlias = Literal["n", "l"]


def get_variant(variant: VariantLiteral) -> int:
    """Returns the base width for the specified variant."""
    variants = {
        "n": 32,
        "l": 64,
    }

    if variant not in variants:
        raise ValueError(
            f"Variant should be one of {list(variants.keys())}, got '{variant}'."
        )

    return variants[variant]


class DiscSubNetHead(BaseHead[Tensor, Tensor]):
    in_channels: list[int] | int
    out_channels: int
    base_channels: int
    task = Tasks.ANOMALY_DETECTION

    def __init__(
        self,
        in_channels: list[int] | int = 6,
        out_channels: int = 2,
        base_channels: int | None = None,
        variant: VariantLiteral = "l",  # Use lowercase variant
        **kwargs,
    ):
        """DiscSubNetHead: A discriminative sub-network that detects and
        segments anomalies in images.

        This model is designed to take an input image and generate a
        mask that highlights anomalies or regions of interest based on
        reconstruction. The encoder extracts relevant features from the
        input, while the decoder generates a mask that identifies areas
        of anomalies by distinguishing between the reconstructed image
        and the input.

        @type in_channels: list[int] | int
        @param in_channels: Number of input channels for the encoder.
            Defaults to 6.
        @type out_channels: int
        @param out_channels: Number of output channels for the decoder.
            Defaults to 2 (for segmentation masks).
        @type base_channels: int
        @param base_channels: The base number of filters used in the
            encoder and decoder blocks. If None, it is determined based
            on the variant.
        @type variant: Literal["n", "l"]
        @param variant: The variant of the DiscSubNetHead to use. "l"
            for large, "n" for nano (lightweight). Defaults to "l".
        """
        super().__init__(**kwargs)

        if isinstance(in_channels, list):
            in_channels = in_channels[0] * 2

        self.base_channels = (
            base_channels
            if base_channels is not None
            else get_variant(variant)
        )

        if variant == "l":
            self.encoder_segment = Encoder(in_channels, self.base_channels)
            self.decoder_segment = Decoder(self.base_channels, out_channels)
        elif variant == "n":
            self.encoder_segment = NanoEncoder(in_channels, self.base_channels)
            self.decoder_segment = NanoDecoder(
                self.base_channels, out_channels
            )

    def forward(self, inputs: list[Tensor]) -> tuple[Tensor, Tensor]:
        """Performs the forward pass through the encoder and decoder."""
        reconstruction, x = inputs
        x = torch.cat([reconstruction, x], dim=1)
        seg_out = self.decoder_segment(*self.encoder_segment(x))

        return seg_out, reconstruction

    def wrap(self, output: tuple[Tensor, Tensor]) -> Packet[Tensor]:
        """Wraps the output into a packet."""
        seg_out, recon = output
        if self.export:
            return {"segmentation": seg_out}
        else:
            seg_out, recon = output
            return {"reconstructed": recon, "segmentation": seg_out}

    def get_custom_head_config(self) -> dict:
        """Returns custom head configuration.

        @rtype: dict
        @return: Custom head configuration.
        """
        return {}
