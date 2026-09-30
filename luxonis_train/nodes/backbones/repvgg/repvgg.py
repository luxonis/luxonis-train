"""The RepVGG backbone, which trains as a multi-branch network and folds
into a plain stack of convolutions in export mode.
"""

from collections import defaultdict

from luxonis_ml.typing import Kwargs
from torch import Tensor, nn
from typeguard import typechecked
from typing_extensions import override

from luxonis_train.nodes.base_node import BaseNode
from luxonis_train.nodes.blocks import GeneralReparameterizableBlock
from luxonis_train.nodes.blocks.utils import forward_gather


class RepVGG(BaseNode):
    r"""RepVGG backbone.

    The backbone is a stem and four stages of
    `GeneralReparameterizableBlock` blocks. Each block sums three
    branches: a :math:`3 \times 3` convolution, a :math:`1 \times 1`
    convolution, and an identity branch when the shape does not change.
    Each branch has a batch norm, and a ``ReLU`` follows the sum. In
    export mode, each block folds its branches into one
    :math:`3 \times 3` convolution. The exported model is then a plain
    stack of convolutions.

    Inputs:
        - ``inputs`` (``Tensor``): :math:`\left[B, C, H, W\right]`

    Outputs:
        - ``features`` (``list[Tensor]``): one per block,
          ``sum(n_blocks)`` total; stages at strides 4, 8, 16, 32

    References:
        - Source: Adapted from `DingXiaoH/RepVGG
          <https://github.com/DingXiaoH/RepVGG>`_ (MIT). Paper: `RepVGG:
          Making VGG-style ConvNets Great Again
          <https://arxiv.org/abs/2101.03697>`_.
        - License: MIT

    Notes:
        `forward` returns the output of every block of the four stages,
        not only of the last block of each stage. The list does not hold
        the stem output. The stem and the first block of each stage have
        a stride of ``2``.

    Variants:
        - ``"A0"``:
            - Default: yes
            - Aliases: None
            - Parameters:
                - ``n_blocks``: ``(2, 4, 14, 1)``
                - ``width_multiplier``: ``(0.75, 0.75, 0.75, 2.5)``
        - ``"A1"``:
            - Default: no
            - Aliases: None
            - Parameters:
                - ``n_blocks``: ``(2, 4, 14, 1)``
                - ``width_multiplier``: ``(1, 1, 1, 2.5)``
        - ``"A2"``:
            - Default: no
            - Aliases: None
            - Parameters:
                - ``n_blocks``: ``(2, 4, 14, 1)``
                - ``width_multiplier``: ``(1.5, 1.5, 1.5, 2.75)``

    Example:
        A node entry in the ``model.nodes`` section of a config:

        .. code-block:: yaml

            - name: RepVGG
              variant: A0

    Compatible with:
        - Attach index: ``-1``, the last output of the input node

    """

    in_channels: int

    @typechecked
    def __init__(
        self,
        n_blocks: tuple[int, int, int, int] = (2, 4, 14, 1),
        width_multiplier: tuple[float, float, float, float] = (
            0.75,
            0.75,
            0.75,
            2.5,
        ),
        override_groups_map: dict[int, int] | None = None,
        use_se: bool = False,
        **kwargs,
    ):
        """Initialize the stem and the four stages.

        Args:
            n_blocks (tuple[int, int, int, int]): The number of blocks in
                each stage. The first block of each stage has a stride
                of ``2``.
            width_multiplier (tuple[float, float, float, float]): The
                channel multipliers ``w`` of the four stages. The stages
                have ``int(64 * w[0])``, ``int(128 * w[1])``,
                ``int(256 * w[2])``, and ``int(512 * w[3])`` output
                channels. The stem has ``min(64, int(64 * w[0]))``
                output channels.
            override_groups_map (dict[int, int] | None): The number of
                groups of the convolutions in each stage, keyed by the
                stage index from ``0`` to ``3``. The key is not a block
                index. A stage without a key, and the stem, use ``1``.
                ``None`` gives ``1`` to all stages.
            use_se (bool): Whether to add a `SqueezeExciteBlock` to the
                stem and to each block. It runs on the sum of the
                branches, before the ``ReLU``, with ``out_channels // 16``
                hidden channels.
            **kwargs (``Any``): Keyword arguments forwarded to
                `BaseNode`.

        """
        super().__init__(**kwargs)

        override_groups_map = defaultdict(lambda: 1, override_groups_map or {})
        self._use_se = use_se

        out_channels = min(64, int(64 * width_multiplier[0]))
        self.stage0 = GeneralReparameterizableBlock(
            in_channels=self.in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            refine_block="se" if use_se else None,
        )

        blocks = []
        in_channels = out_channels
        for i in range(4):
            out_channels = int(2**i * 64 * width_multiplier[i])
            blocks.extend(
                self._make_stage(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    strides=[2] + [1] * (n_blocks[i] - 1),
                    groups=override_groups_map[i],
                )
            )
            in_channels = out_channels

        self.blocks = nn.ModuleList(blocks)

    def forward(self, inputs: Tensor) -> list[Tensor]:
        """Run the stem and the four stages on a batch of images.

        Args:
            inputs (``Tensor``): The input images, of shape
                ``[B, C, H, W]``.

        Returns:
            ``list[Tensor]``: The output of each block of the four
            stages, in order. The list holds ``sum(n_blocks)`` tensors
            and not the stem output. The blocks of stage ``i``, from
            ``0`` to ``3``, have a stride of ``2 ** (i + 2)``.

        Example:
            The indices ``1``, ``5``, ``19``, and ``20`` select the last
            block of each stage of ``"A0"``.

            >>> import torch
            >>> from torch import Size
            >>> from luxonis_train.nodes.backbones import RepVGG
            >>> shapes = [{"features": [Size([1, 3, 64, 64])]}]
            >>> node = RepVGG(input_shapes=shapes, variant="A0")
            >>> features = node(torch.zeros(1, 3, 64, 64))
            >>> len(features)
            21
            >>> [tuple(features[i].shape) for i in (1, 5, 19, 20)]
            [(1, 48, 16, 16), (1, 96, 8, 8), (1, 192, 4, 4), (1, 1280, 2, 2)]

        """
        return forward_gather(self.stage0(inputs), self.blocks)

    @staticmethod
    @override
    def get_variants() -> tuple[str, dict[str, Kwargs]]:
        """Return the default variant name and the three RepVGG variants.

        All variants set ``n_blocks`` to ``(2, 4, 14, 1)``. They differ in
        ``width_multiplier``. No variant sets ``override_groups_map`` or
        ``use_se``. Each call builds new dictionaries.

        Returns:
            ``tuple[str, dict[str, Kwargs]]``: The name of the default
            variant, ``"A0"``, and a dictionary that maps ``"A0"``,
            ``"A1"``, and ``"A2"`` to their constructor arguments.

        Example:
            >>> from luxonis_train.nodes.backbones import RepVGG
            >>> default, variants = RepVGG.get_variants()
            >>> default, sorted(variants)
            ('A0', ['A0', 'A1', 'A2'])
            >>> variants["A2"]["width_multiplier"]
            (1.5, 1.5, 1.5, 2.75)

        """
        return "A0", {
            "A0": {
                "n_blocks": (2, 4, 14, 1),
                "width_multiplier": (0.75, 0.75, 0.75, 2.5),
            },
            "A1": {
                "n_blocks": (2, 4, 14, 1),
                "width_multiplier": (1, 1, 1, 2.5),
            },
            "A2": {
                "n_blocks": (2, 4, 14, 1),
                "width_multiplier": (1.5, 1.5, 1.5, 2.75),
            },
        }

    def _make_stage(
        self,
        in_channels: int,
        out_channels: int,
        strides: list[int],
        groups: int,
    ) -> list[nn.Module]:
        stage = []
        for stride in strides:
            stage.append(
                GeneralReparameterizableBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    groups=groups,
                    refine_block="se" if self._use_se else None,
                )
            )
            in_channels = out_channels
        return stage
