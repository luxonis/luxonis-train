"""The DRAEM discriminative subnetwork, which segments the anomaly from
the image and its reconstruction.
"""

import torch
from luxonis_ml.typing import Kwargs
from torch import Tensor
from typing_extensions import override

from luxonis_train.nodes.blocks import UNetDecoder, UNetEncoder
from luxonis_train.nodes.heads import BaseHead
from luxonis_train.tasks import Tasks
from luxonis_train.typing import Packet


class DiscSubNetHead(BaseHead):
    r"""Discriminative anomaly segmentation head.

    Inputs:
        - ``reconstruction`` (``Tensor``): :math:`\left[B, C, H,
          W\right]`
        - ``original`` (``Tensor``): :math:`\left[B, C, H, W\right]`

    Outputs:
        - train, eval:

          - ``segmentation`` (``Tensor``): :math:`\left[B, K, H,
            W\right]` logits, where :math:`K` is ``out_channels``
          - ``reconstruction`` (``Tensor``): :math:`\left[B, C, H,
            W\right]`, the input unchanged

        - export:

          - ``segmentation`` (``Tensor``): :math:`\left[B, K, H,
            W\right]` logits

    References:
        - Source: This project.
        - License: Apache-2.0 (this project)

    Notes:
        The head reads the two outputs of `RecSubNet`. It concatenates
        ``reconstruction`` and ``original`` along the channel axis, in
        this order. A U-Net then maps the :math:`2C` channels to
        :math:`K` channels of anomaly logits:

        - The encoder has one level for each value in
          ``width_multipliers``, and one more level that repeats the
          last value. Each level after the first starts with a ``2x2``
          max pool.
        - The decoder upsamples the map one level at a time. At each
          level, it concatenates the encoder map of the same size.

        **The height and the width must be multiples of** :math:`2^L`,
        where :math:`L` is the length of ``width_multipliers``. This is
        ``4`` for ``"n"`` and ``32`` for ``"l"``. With other sizes, an
        upsampled map does not fit its encoder map, and the
        concatenation raises ``RuntimeError``. The constructor reads
        :math:`C` from the two input
        shapes. When the shapes differ, it raises ``RuntimeError``.

        `ReconstructionSegmentationLoss` reads both outputs. Export mode
        removes ``reconstruction``.

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

    See Also:
        - `DRAEM - A discriminatively trained reconstruction embedding
          for surface anomaly detection
          <https://arxiv.org/abs/2108.07610>`_

    Example:
        A node entry in the ``model.nodes`` section of a config:

        .. code-block:: yaml

            - name: DiscSubNetHead
              inputs: [RecSubNet]
              variant: n

    Compatible with:
        - Attach index: ``-1``, the last output of the input node
        - Required labels:

          - ``original_segmentation``
          - ``segmentation``

        - Used by: `AnomalyDetectionModel`
        - Losses: `ReconstructionSegmentationLoss`
        - Metrics:

          - `Accuracy`
          - `F1Score`
          - `JaccardIndex`
          - `Precision`
          - `Recall`

        - Visualizers: `SegmentationVisualizer`

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
        """Build the U-Net encoder and decoder.

        The encoder reads ``2 * in_channels`` channels, because
        `forward` concatenates the two inputs. Each encoder level is a
        `ConvStack` of two ``3x3`` convolutions. Each decoder level
        doubles the size with bilinear interpolation and a ``3x3``
        `ConvBlock`. Then it runs a `ConvStack` on the result and the
        encoder map. A last ``3x3`` convolution maps ``base_channels``
        to ``out_channels``.

        Args:
            base_channels (int): The width that the multipliers scale.
                The last decoder level always has this number of
                channels.
            width_multipliers (list[float]): One multiplier for each
                encoder level. A level has ``int(base_channels * m)``
                channels. The encoder adds one more level with the last
                multiplier. For example, ``32`` and ``[1, 1.1]`` give the
                levels ``32``, ``35``, and ``35``.
            out_channels (int): The number of channels of the anomaly
                logits. The ``"segmentation"`` label of the anomaly
                detection task has two channels, so
                `ReconstructionSegmentationLoss` needs ``2``.
            **kwargs (``Any``): Keyword arguments for `BaseNode`. They
                must hold ``input_shapes`` or ``in_sizes``.

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
        """Segment the anomalies from the image and its reconstruction.

        The method concatenates ``reconstruction`` and ``original`` along
        the channel axis and runs the U-Net on the result.

        Args:
            reconstruction (``Tensor``): The images that `RecSubNet`
                rebuilds, of shape ``[B, C, H, W]``. ``H`` and ``W`` must
                be multiples of ``2 ** len(width_multipliers)``. Otherwise,
                a concatenation in the U-Net raises ``RuntimeError``.
            original (``Tensor``): The input images, of the same shape.

        Returns:
            ``Packet[Tensor]``: The anomaly logits of shape
            ``[B, out_channels, H, W]`` under the ``"segmentation"`` key.
            Outside export mode, the packet also holds ``reconstruction``
            unchanged under the ``"reconstruction"`` key.

        Example:
            >>> import torch
            >>> from torch import Size
            >>> from luxonis_train.nodes import DiscSubNetHead
            >>> shape = Size([1, 3, 8, 8])
            >>> head = DiscSubNetHead(
            ...     variant="n",
            ...     input_shapes=[
            ...         {"reconstruction": shape, "original": shape}
            ...     ],
            ... )
            >>> image = torch.zeros(shape)
            >>> packet = head.run(
            ...     [{"reconstruction": image, "original": image}]
            ... )
            >>> sorted(packet), packet["segmentation"].shape
            (['reconstruction', 'segmentation'], torch.Size([1, 2, 8, 8]))
            >>> head.export = True
            >>> sorted(head(image, image))
            ['segmentation']

        """
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
        """Return the default variant ``"n"`` and the two variants.

        Both variants set ``base_channels`` and ``width_multipliers``.
        ``"n"`` has three encoder levels and needs a height and a width
        that are multiples of ``4``. ``"l"`` has six encoder levels, up
        to ``512`` channels, and needs multiples of ``32``.

        Returns:
            ``tuple[str, dict[str, Kwargs]]``: The name ``"n"``, and a
            dictionary that maps ``"n"`` and ``"l"`` to their
            constructor keyword arguments.

        """
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
