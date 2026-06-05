from luxonis_ml.typing import Kwargs
from torch import Tensor, nn
from typeguard import typechecked
from typing_extensions import override

from luxonis_train.nodes.base_node import BaseNode
from luxonis_train.nodes.blocks import (
    GeneralReparametrizableBlock,
    SqueezeExciteBlock,
)
from luxonis_train.nodes.blocks.utils import forward_gather


class MobileOne(BaseNode):
    """MobileOne efficient CNN backbone for mobile devices.

    MobileOne uses simple convolutional stages and scaled channel widths to
    provide latency-focused feature extraction for mobile deployments.

    Metadata:
        - Node type: backbone
        - Registry name: ``MobileOne``
        - Task: None
        - Attach index: ``-1``
        - Inputs: ``features`` tensor
        - Outputs: ``features`` list of tensors

    Provenance:
        - Source: ``apple/ml-mobileone``
        - License: Apple
        - Implementation notes: Local MobileOne-style staged backbone with
          configurable width multipliers and optional squeeze-excitation.

    Variants:
        - ``"s0"``:
            - Default: yes
            - Aliases: None
            - Parameters:
                - ``width_multipliers``: ``(0.75, 1.0, 1.0, 2.0)``
                - ``n_conv_branches``: ``4``
                - ``use_se``: ``False``
        - ``"s1"``:
            - Default: no
            - Aliases: None
            - Parameters:
                - ``width_multipliers``: ``(1.5, 1.5, 2.0, 2.5)``
                - ``n_conv_branches``: ``4``
                - ``use_se``: ``False``
        - ``"s2"``:
            - Default: no
            - Aliases: None
            - Parameters:
                - ``width_multipliers``: ``(1.5, 2.0, 2.5, 4.0)``
                - ``n_conv_branches``: ``4``
                - ``use_se``: ``False``
        - ``"s3"``:
            - Default: no
            - Aliases: None
            - Parameters:
                - ``width_multipliers``: ``(2.0, 2.5, 3.0, 4.0)``
                - ``n_conv_branches``: ``4``
                - ``use_se``: ``False``
        - ``"s4"``:
            - Default: no
            - Aliases: None
            - Parameters:
                - ``width_multipliers``: ``(3.0, 3.5, 3.5, 4.0)``
                - ``n_conv_branches``: ``4``
                - ``use_se``: ``True``

    """

    in_channels: int

    @typechecked
    def __init__(
        self,
        width_multipliers: tuple[float, float, float, float] = (
            0.75,
            1.0,
            1.0,
            2.0,
        ),
        n_conv_branches: int = 4,
        use_se: bool = False,
        **kwargs,
    ):
        """Initialize the MobileOne backbone.

        Args:
            width_multipliers (tuple[float, float, float, float]): Width multipliers for each stage.
            n_conv_branches (int): Number of linear convolution branches in MobileOne block.
            use_se (bool): Whether to use ``Squeeze-and-Excitation`` blocks in the network. Default is ``False``.
            **kwargs (``Any``): Keyword arguments forwarded to the parent class.

        """
        super().__init__(**kwargs)

        self.n_blocks_per_stage = [2, 8, 10, 1]
        self.n_conv_branches = n_conv_branches

        self._in_channels = min(64, int(64 * width_multipliers[0]))

        self.stages = nn.ModuleList(
            [
                GeneralReparametrizableBlock(
                    in_channels=self.in_channels,
                    out_channels=self._in_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                self._make_stage(
                    int(64 * width_multipliers[0]),
                    self.n_blocks_per_stage[0],
                    n_se_blocks=0,
                ),
                self._make_stage(
                    int(128 * width_multipliers[1]),
                    self.n_blocks_per_stage[1],
                    n_se_blocks=0,
                ),
                self._make_stage(
                    int(256 * width_multipliers[2]),
                    self.n_blocks_per_stage[2],
                    n_se_blocks=self.n_blocks_per_stage[2] // 2
                    if use_se
                    else 0,
                ),
                self._make_stage(
                    int(512 * width_multipliers[3]),
                    self.n_blocks_per_stage[3],
                    n_se_blocks=self.n_blocks_per_stage[3] if use_se else 0,
                ),
            ]
        )

    def forward(self, inputs: Tensor) -> list[Tensor]:
        return forward_gather(inputs, self.stages)

    @override
    @staticmethod
    def get_variants() -> tuple[str, dict[str, Kwargs]]:
        return "s0", {
            "s0": {
                "width_multipliers": (0.75, 1.0, 1.0, 2.0),
                "n_conv_branches": 4,
                "use_se": False,
            },
            "s1": {
                "width_multipliers": (1.5, 1.5, 2.0, 2.5),
                "use_se": False,
            },
            "s2": {
                "width_multipliers": (1.5, 2.0, 2.5, 4.0),
                "use_se": False,
            },
            "s3": {
                "width_multipliers": (2.0, 2.5, 3.0, 4.0),
                "use_se": False,
            },
            "s4": {
                "width_multipliers": (3.0, 3.5, 3.5, 4.0),
                "use_se": True,
            },
        }

    def _make_stage(
        self, out_channels: int, n_blocks: int, n_se_blocks: int
    ) -> nn.Sequential:
        """Build a stage of MobileOne model.

        Args:
            out_channels (int): Number of output channels.
            n_blocks (int): Number of blocks in this stage.
            n_se_blocks (int): Number of SE blocks in this stage.

        Returns:
            ``nn.Sequential``: A stage of MobileOne model.

        """
        # Get strides for all layers
        strides = [2] + [1] * (n_blocks - 1)
        blocks: list[nn.Module] = []
        for ix, stride in enumerate(strides):
            if n_se_blocks > n_blocks:
                raise ValueError(
                    "Number of SE blocks cannot exceed number of layers."
                )
            if ix >= (n_blocks - n_se_blocks):
                refine_block = SqueezeExciteBlock(
                    in_channels=self._in_channels,
                    intermediate_channels=self._in_channels // 16,
                )
            else:
                refine_block = nn.Identity()

            # Depthwise conv
            blocks.append(
                GeneralReparametrizableBlock(
                    in_channels=self._in_channels,
                    out_channels=self._in_channels,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    groups=self._in_channels,
                    n_branches=self.n_conv_branches,
                    refine_block=refine_block,
                    scale_layer_padding=0,
                )
            )
            # Pointwise conv
            blocks.append(
                GeneralReparametrizableBlock(
                    in_channels=self._in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    groups=1,
                    n_branches=self.n_conv_branches,
                    refine_block=refine_block,
                    use_scale_layer=False,
                )
            )
            self._in_channels = out_channels
        return nn.Sequential(*blocks)
