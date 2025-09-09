import torch
from luxonis_ml.typing import Kwargs, Params
from torch import Tensor
from typing_extensions import override

from luxonis_train.nodes.blocks import UNetDecoder, UNetEncoder
from luxonis_train.nodes.heads import BaseHead
from luxonis_train.tasks import Tasks
from luxonis_train.typing import Packet


class DiscSubNetHead(BaseHead):
    task = Tasks.ANOMALY_DETECTION

    in_channels: int
    base_channels: int

    attach_index = -1

    def __init__(
        self,
        base_channels: int,
        width_multipliers: list[float],
        out_channels: int = 2,
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

        @type out_channels: int
        @param out_channels: Number of output channels for the decoder.
            Defaults to 2 (for segmentation masks).

        @type base_channels: int
        @param base_channels: The base number of filters used in the encoder and decoder blocks. If None, it is determined based on the variant.
        """
        super().__init__(**kwargs)

        self.encoder_segment = UNetEncoder(
            self.in_channels * 2, base_channels, width_multipliers
        )
        self.decoder_segment = UNetDecoder(
            base_channels, out_channels, width_multipliers
        )

    def forward(
        self, reconstruction: Tensor, original: Tensor
    ) -> Packet[Tensor]:
        """Performs the forward pass through the encoder and decoder."""
        x = torch.cat([reconstruction, original], dim=1)
        seg_out = self.decoder_segment(self.encoder_segment(x))

        if self.export:
            return {self.task.main_output: seg_out}

        return {
            self.task.main_output: seg_out,
            "reconstruction": reconstruction,
        }

    @override
    def get_custom_head_config(self) -> Params:
        """Returns custom head configuration.

        @rtype: dict
        @return: Custom head configuration.
        """
        return {}

    @staticmethod
    @override
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
