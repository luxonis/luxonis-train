"""The EfficientViT backbone.

EfficientViT uses a ReLU linear attention instead of a softmax
attention. Thus the cost of the attention grows linearly with the number
of positions in the feature map.

"""

from luxonis_ml.typing import Kwargs
from torch import Tensor, nn
from typeguard import typechecked
from typing_extensions import override

from luxonis_train.nodes.base_node import BaseNode
from luxonis_train.nodes.blocks import ConvBlock
from luxonis_train.variants import add_variant_aliases

from .blocks import (
    DepthWiseSeparableConv,
    EfficientViTBlock,
    MobileBottleneckBlock,
)


class EfficientViT(BaseNode):
    r"""EfficientViT backbone with multi-scale linear attention.

    The backbone has one stage for each entry of ``width_list``. Stage
    ``i`` has ``width_list[i]`` output channels. Each stage halves the
    height and the width of its input. The stages have these blocks:

    - Stage ``0``: a ``3x3`` `ConvBlock` with a stride of ``2``, then
      ``depth_list[0]`` `DepthWiseSeparableConv` blocks.
    - Stages ``1`` and ``2``: ``depth_list[i]`` `MobileBottleneckBlock`
      blocks. The first block has a stride of ``2``.
    - Stage ``3`` and each later stage: a `MobileBottleneckBlock` with a
      stride of ``2``, then ``depth_list[i]`` `EfficientViTBlock`
      blocks. Only these stages use attention.

    Every block with a stride of ``1`` adds its input to its output.

    Inputs:
        - ``inputs`` (``Tensor``): :math:`\left[B, C, H, W\right]`

    Outputs:
        - ``features`` (``list[Tensor]``): one tensor for each stage;
          stage ``i`` has ``width_list[i]`` channels and the stride
          :math:`2^{i+1}`, so strides 2, 4, 8, 16, 32 for the variants

    References:
        - Source: Reimplemented from `EfficientViT: Multi-Scale Linear
          Attention for High-Resolution Dense Prediction
          <https://arxiv.org/abs/2205.14756>`_.
        - License: Apache-2.0 (this project)

    Notes:
        A stage rounds an odd height or width up when it halves it. A
        ``depth_list[1]`` or ``depth_list[2]`` of ``0`` gives an empty
        stage. That stage returns its input, so its output has the
        channels and the stride of the stage before it. From index
        ``3``, ``width_list[i]`` must be at least ``dim`` when
        ``depth_list[i]`` is not ``0``. Otherwise the attention blocks
        of the stage get no heads, and the constructor raises
        ``ValueError``.

    Variants:
        - ``"n"``:
            - Default: yes
            - Aliases: ``"nano"``
            - Parameters:
                - ``width_list``: ``[8, 16, 32, 64, 128]``
                - ``depth_list``: ``[1, 2, 2, 2, 2]``
                - ``dim``: ``16``
        - ``"s"``:
            - Default: no
            - Aliases: ``"small"``
            - Parameters:
                - ``width_list``: ``[16, 32, 64, 128, 256]``
                - ``depth_list``: ``[1, 2, 3, 3, 4]``
                - ``dim``: ``16``
        - ``"m"``:
            - Default: no
            - Aliases: ``"medium"``
            - Parameters:
                - ``width_list``: ``[24, 48, 96, 192, 384]``
                - ``depth_list``: ``[1, 3, 4, 4, 6]``
                - ``dim``: ``32``
        - ``"l"``:
            - Default: no
            - Aliases: ``"large"``
            - Parameters:
                - ``width_list``: ``[32, 64, 128, 256, 512]``
                - ``depth_list``: ``[1, 4, 6, 6, 9]``
                - ``dim``: ``32``

    Example:
        A node entry in the ``model.nodes`` section of a config:

        .. code-block:: yaml

            - name: EfficientViT
              variant: n

    Compatible with:
        - Attach index: ``-1``, the last output of the input node

    """

    in_channels: int

    @typechecked
    def __init__(
        self,
        width_list: list[int] | None = None,
        depth_list: list[int] | None = None,
        dim: int = 16,
        expand_ratio: int = 4,
        **kwargs,
    ):
        """Build the stages of the backbone.

        The constructor reads `BaseNode.in_channels`, so the call must
        give ``input_shapes`` or ``in_sizes``. Without them, that
        property raises ``RuntimeError``.

        Args:
            width_list (list[int] | None): Number of output channels of
                each stage. The length sets the number of stages. ``None``
                or an empty list selects ``[8, 16, 32, 64, 128]``.
            depth_list (list[int] | None): Number of repeated blocks of
                each stage. Stage ``0`` has this number of
                `DepthWiseSeparableConv` blocks after its stem
                convolution. Stages ``1`` and ``2`` have this number of
                `MobileBottleneckBlock` blocks. A later stage has this
                number of `EfficientViTBlock` blocks after its first
                `MobileBottleneckBlock`. ``None`` or an empty list
                selects ``[1, 2, 2, 2, 2]``.
            dim (int): Number of channels of the query, the key, and the
                value of each attention head in the `EfficientViTBlock`
                blocks. Defaults to ``16``.
            expand_ratio (int): Channel expansion factor of every
                `MobileBottleneckBlock`. This includes the
                `MobileBottleneckBlock` of each `EfficientViTBlock`.
                Defaults to ``4``.
            **kwargs (``Any``): Keyword arguments forwarded to
                `BaseNode`.

        Raises:
            ValueError: When ``width_list`` and ``depth_list`` have
                different lengths. Also when a stage from index ``3``
                has `EfficientViTBlock` blocks and ``width_list[i]`` is
                smaller than ``dim``.

        """
        super().__init__(**kwargs)
        width_list = width_list or [8, 16, 32, 64, 128]
        depth_list = depth_list or [1, 2, 2, 2, 2]

        self.feature_extractor = nn.ModuleList(
            [
                ConvBlock(
                    in_channels=self.in_channels,
                    out_channels=width_list[0],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    activation=nn.Hardswish(),
                )
            ]
        )
        for _ in range(depth_list[0]):
            block = DepthWiseSeparableConv(
                in_channels=width_list[0],
                out_channels=width_list[0],
                stride=1,
                depthwise_activation=nn.Hardswish(),
                use_residual=True,
            )
            self.feature_extractor.append(block)

        in_channels = width_list[0]
        self.encoder_blocks = nn.ModuleList()
        for w, d in zip(width_list[1:3], depth_list[1:3], strict=True):
            encoder_blocks = nn.ModuleList()
            for i in range(d):
                stride = 2 if i == 0 else 1
                block = MobileBottleneckBlock(
                    in_channels=in_channels,
                    out_channels=w,
                    stride=stride,
                    expand_ratio=expand_ratio,
                    use_norm=[True, True, True],
                    activation=[
                        nn.Hardswish(),
                        nn.Hardswish(),
                        nn.Identity(),
                    ],
                    use_bias=[False, False, False],
                    use_residual=stride == 1,
                )
                encoder_blocks.append(block)
                in_channels = w
            self.encoder_blocks.append(encoder_blocks)

        for w, d in zip(width_list[3:], depth_list[3:], strict=True):
            encoder_blocks = nn.ModuleList()
            block = MobileBottleneckBlock(
                in_channels=in_channels,
                out_channels=w,
                stride=2,
                expand_ratio=expand_ratio,
                activation=[
                    nn.Hardswish(),
                    nn.Hardswish(),
                    nn.Identity(),
                ],
                use_norm=[False, False, True],
                use_residual=False,
            )
            encoder_blocks.append(block)
            in_channels = w

            for _ in range(d):
                encoder_blocks.append(
                    EfficientViTBlock(
                        n_channels=in_channels,
                        head_dim=dim,
                        expansion_factor=expand_ratio,
                    )
                )
            self.encoder_blocks.append(encoder_blocks)

    def forward(self, x: Tensor) -> list[Tensor]:
        r"""Run the stages and return the output of each stage.

        Args:
            x (``Tensor``): Input of shape ``[B, C, H, W]``, where ``C``
                is `BaseNode.in_channels`.

        Returns:
            ``list[Tensor]``: One tensor for each stage, in stage order.
            When no stage is empty, stage ``i`` gives the shape
            ``[B, width_list[i], H_i, W_i]``, with
            :math:`H_i = \lceil H / 2^{i+1} \rceil` and
            :math:`W_i = \lceil W / 2^{i+1} \rceil`. An empty stage
            repeats the tensor of the stage before it.

        Example:
            >>> import torch
            >>> from torch import Size
            >>> from luxonis_train.nodes import EfficientViT
            >>> shapes = [{"features": [Size([1, 3, 64, 64])]}]
            >>> node = EfficientViT(variant="n", input_shapes=shapes)
            >>> features = node(torch.zeros(1, 3, 64, 64))
            >>> [tuple(feature.shape) for feature in features]
            [(1, 8, 32, 32), (1, 16, 16, 16), (1, 32, 8, 8),
             (1, 64, 4, 4), (1, 128, 2, 2)]

        """
        outputs = []
        for block in self.feature_extractor:
            x = block(x)
        outputs.append(x)
        for encoder_blocks in self.encoder_blocks:
            for block in encoder_blocks:  # type: ignore
                x = block(x)
            outputs.append(x)
        return outputs

    @override
    @staticmethod
    def get_variants() -> tuple[str, dict[str, Kwargs]]:
        """Return the default variant name and the EfficientViT variants.

        The variants ``"n"``, ``"s"``, ``"m"``, and ``"l"`` set
        ``width_list``, ``depth_list``, and ``dim``. No variant sets
        ``expand_ratio``. The aliases ``"nano"``, ``"small"``,
        ``"medium"``, and ``"large"`` map to the same dictionary objects
        as their variants. Each call builds new dictionaries.

        Returns:
            ``tuple[str, dict[str, Kwargs]]``: The name ``"n"``, and a
            dictionary that maps each variant name and alias to its
            constructor keyword arguments.

        Example:
            >>> from luxonis_train.nodes import EfficientViT
            >>> default, variants = EfficientViT.get_variants()
            >>> default, variants["small"]["width_list"]
            ('n', [16, 32, 64, 128, 256])
            >>> sorted(variants)
            ['l', 'large', 'm', 'medium', 'n', 'nano', 's', 'small']

        """
        return "n", add_variant_aliases(
            {
                "n": {
                    "width_list": [8, 16, 32, 64, 128],
                    "depth_list": [1, 2, 2, 2, 2],
                    "dim": 16,
                },
                "s": {
                    "width_list": [16, 32, 64, 128, 256],
                    "depth_list": [1, 2, 3, 3, 4],
                    "dim": 16,
                },
                "m": {
                    "width_list": [24, 48, 96, 192, 384],
                    "depth_list": [1, 3, 4, 4, 6],
                    "dim": 32,
                },
                "l": {
                    "width_list": [32, 64, 128, 256, 512],
                    "depth_list": [1, 4, 6, 6, 9],
                    "dim": 32,
                },
            }
        )
