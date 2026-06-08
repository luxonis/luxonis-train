import torch
from luxonis_ml.typing import Kwargs
from torch import Tensor
from typing_extensions import override

from luxonis_train.nodes.blocks import UNetDecoder, UNetEncoder
from luxonis_train.nodes.heads import BaseHead
from luxonis_train.tasks import Tasks
from luxonis_train.typing import Packet


class DiscSubNetHead(BaseHead):
    """Discriminative anomaly segmentation head.

    Metadata:
        - Node type: head
        - Registry name: ``DiscSubNetHead``
        - Task: anomaly_detection
        - Attach index: ``-1``
        - Inputs: reconstruction tensor and original tensor
        - Outputs: export returns ``segmentation``; train and eval return
          ``segmentation`` and ``reconstruction``.

    Provenance:
        - Source: Internal
        - License: Project license
        - Implementation notes: Concatenates reconstruction and original
          tensors, then applies a U-Net encoder-decoder segmentation
          network.

    Variants:
        - ``"n"``:
            - Default: yes
            - Aliases: None
            - Parameters:
                - ``base_channels``: ``32``
                - ``width_multipliers``: ``[1, 1.1]``
        - ``"l"``:
            - Default: no
            - Aliases: None
            - Parameters:
                - ``base_channels``: ``64``
                - ``width_multipliers``: ``[1, 2, 4, 8, 8]``

    """

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
        """Discriminative sub-network for segmenting anomalies.

        This model is designed to take an input image and generate a
        mask that highlights anomalies or regions of interest based on
        reconstruction. The encoder extracts relevant features from the
        input, while the decoder generates a mask that identifies areas
        of anomalies by distinguishing between the reconstructed image
        and the input.

        Args:
            base_channels (int): The base number of filters used in the encoder and decoder blocks.
            width_multipliers (list[float]): A list of multipliers that determine the number of filters in each block of the encoder and decoder. Each multiplier is applied to the base_channels to calculate the number of filters for that block. For example, if base_channels is 32 and width_multipliers is [1, 2], the first block will have 32 filters and the second block will have 64 filters.
            out_channels (int): Number of output channels for the decoder. Defaults to 2 (for segmentation masks).
            **kwargs (``Any``): Keyword arguments forwarded to the parent class.

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
        """Perform the forward pass through the encoder and decoder."""
        x = torch.cat([reconstruction, original], dim=1)
        seg_out = self.decoder_segment(self.encoder_segment(x))

        if self.export:
            return {self.task.main_output: seg_out}

        return {
            self.task.main_output: seg_out,
            "reconstruction": reconstruction,
        }

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
