"""The DRAEM reconstruction subnetwork, which rebuilds an image without
its anomalies so the head can compare the two.
"""

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
    r"""RecSubNet reconstruction backbone of DRAEM anomaly detection.

    RecSubNet is a `SimpleEncoder` and a `SimpleDecoder`. It reads an
    image that can contain anomalies. It learns to reconstruct the image
    without them, for example with `ReconstructionSegmentationLoss`. The
    node returns the reconstruction and the unchanged input, so that
    `DiscSubNetHead` can compare the two. The node does not return
    feature maps for other heads.

    Inputs:
        - ``inputs`` (``Tensor``): :math:`\left[B, C, H, W\right]`

    Outputs:
        - ``reconstruction`` (``Tensor``): :math:`\left[B, out_{channels},
          H', W'\right]`
        - ``original`` (``Tensor``): :math:`\left[B, C, H, W\right]`,
          the input unchanged

    References:
        - Source: This project. Paper: `DRAEM - A Discriminatively
          Trained Reconstruction Embedding for Surface Anomaly Detection
          <https://arxiv.org/abs/2108.07610>`_.
        - License: Apache-2.0 (this project)

    Notes:
        The encoder halves the height and the width once for each value
        of ``width_multipliers``. The decoder doubles them the same
        number of times. Thus ``H'`` and ``W'`` are ``H`` and ``W``
        rounded down to a multiple of ``2 ** len(width_multipliers)``.
        This is ``4`` for ``"n"`` and ``16`` for ``"l"``. The
        reconstruction has the input size only when ``H`` and ``W`` are
        such multiples.

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

    Example:
        A node entry in the ``model.nodes`` section of a config:

        .. code-block:: yaml

            - name: RecSubNet
              variant: l

    Compatible with:
        - Attach index: ``-1``, the last output of the input node
        - Used by: `AnomalyDetectionModel`

    """

    in_channels: int

    def __init__(
        self,
        base_channels: int = 128,
        width_multipliers: list[float] | None = None,
        out_channels: int = 3,
        **kwargs,
    ):
        r"""Initialize the encoder and the decoder.

        The encoder has ``len(width_multipliers) + 1`` stages. Each stage
        has two :math:`3 \times 3` convolutions with batch norm and
        ``ReLU``. Each stage after the first starts with a
        :math:`2 \times 2` max pooling. The decoder has
        ``len(width_multipliers)`` stages. Each stage doubles the height
        and the width with bilinear upsampling. Then it applies three
        :math:`3 \times 3` convolutions with batch norm and ``ReLU``. A
        last :math:`3 \times 3` convolution without an activation gives
        the reconstruction.

        Args:
            base_channels (int): The base width. Encoder stage ``i`` has
                ``int(base_channels * width_multipliers[i])`` channels,
                and the last encoder stage repeats the last multiplier.
                The decoder uses the multipliers in reverse order. Its
                last stage has ``base_channels`` channels.
            width_multipliers (list[float] | None): The channel
                multipliers of the encoder stages. ``None`` or an empty
                list selects ``[1, 2, 4, 8]``.
            out_channels (int): The number of channels of the
                reconstruction.
            **kwargs (``Any``): Keyword arguments forwarded to
                `BaseNode`.

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
        """Reconstruct a batch of images.

        Args:
            x (``Tensor``): The input images, of shape ``[B, C, H, W]``.

        Returns:
            ``Packet[Tensor]``: A packet with two keys.
            ``"reconstruction"`` holds the decoder output, of shape
            ``[B, out_channels, H', W']``. ``H'`` and ``W'`` are ``H``
            and ``W`` rounded down to a multiple of
            ``2 ** len(width_multipliers)``. ``"original"`` holds ``x``
            unchanged.

        Example:
            >>> import torch
            >>> from torch import Size
            >>> from luxonis_train.nodes.backbones import RecSubNet
            >>> shapes = [{"features": [Size([1, 3, 32, 32])]}]
            >>> node = RecSubNet(input_shapes=shapes, variant="n")
            >>> packet = node(torch.zeros(1, 3, 32, 32))
            >>> tuple(packet["reconstruction"].shape)
            (1, 3, 32, 32)

            The ``"n"`` variant rounds the size down to a multiple of
            ``4``:

            >>> packet = node(torch.zeros(1, 3, 30, 30))
            >>> {key: tuple(value.shape) for key, value in packet.items()}
            {'reconstruction': (1, 3, 28, 28), 'original': (1, 3, 30, 30)}

        """
        return {
            "reconstruction": self.decoder(self.encoder(x)),
            "original": x,
        }

    @staticmethod
    @override
    def get_variants() -> tuple[str, dict[str, Kwargs]]:
        """Return the default variant name and the two RecSubNet variants.

        Both variants set ``base_channels`` and ``width_multipliers``.
        ``"n"`` has two multipliers, so it keeps sizes that are multiples
        of ``4``. ``"l"`` has four multipliers, so it keeps sizes that
        are multiples of ``16``. Each call builds new dictionaries.

        Returns:
            ``tuple[str, dict[str, Kwargs]]``: The name of the default
            variant, ``"l"``, and a dictionary that maps ``"n"`` and
            ``"l"`` to their constructor arguments.

        Example:
            >>> from luxonis_train.nodes.backbones import RecSubNet
            >>> default, variants = RecSubNet.get_variants()
            >>> default, variants["n"]
            ('l', {'base_channels': 64, 'width_multipliers': [1, 1.1]})

        """
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
