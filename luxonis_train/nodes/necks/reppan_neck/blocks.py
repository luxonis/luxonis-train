"""The upsampling and downsampling blocks of the RepPAN neck."""

from abc import ABC

import torch
from torch import Tensor, nn

from luxonis_train.nodes.blocks import (
    BlockRepeater,
    ConvBlock,
    CSPStackRepBlock,
    GeneralReparameterizableBlock,
)


class PANUpBlockBase(ABC, nn.Module):
    """Base class of the top-down fusion steps of `RepPANNeck`.

    A ``1x1`` `ConvBlock` with batch norm and ReLU maps the coarse input
    to ``out_channels``. A ``2x2`` transposed convolution with stride
    ``2`` then doubles the height and the width. The block concatenates
    the result with the finer input along the channel axis and runs
    ``encode_block`` on it. A subclass selects ``encode_block``.

    """

    def __init__(
        self, in_channels: int, out_channels: int, encode_block: nn.Module
    ):
        """Build the ``1x1`` convolution and the upsampling layer.

        Args:
            in_channels (int): Number of channels of the coarse input.
            out_channels (int): Number of channels after the ``1x1``
                convolution. The upsampling layer keeps this number.
            encode_block (``nn.Module``): Block that runs on the
                concatenation. Its input has ``out_channels`` plus the
                channels of the finer input.

        """
        super().__init__()

        self.conv = ConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
        )
        self.upsample = nn.ConvTranspose2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=2,
            stride=2,
            bias=True,
        )
        self.encode_block = encode_block

    def forward(self, x0: Tensor, x1: Tensor) -> tuple[Tensor, Tensor]:
        """Upsample the coarse map and fuse it with the finer map.

        Args:
            x0 (``Tensor``): Coarse map of shape ``[B, in_channels, H, W]``.
            x1 (``Tensor``): Finer map of shape ``[B, C1, 2 * H, 2 * W]``.
                In the subclasses, ``C1`` is ``in_channels_next``.

        Returns:
            ``tuple[Tensor, Tensor]``: The tuple ``(conv_out, out)``.
            ``conv_out`` is the output of the ``1x1`` convolution, of
            shape ``[B, out_channels, H, W]``. `RepPANNeck` gives it to a
            bottom-up step as the lateral input. ``out`` is the output of
            ``encode_block``. In the subclasses, ``out`` has the shape
            ``[B, out_channels, 2 * H, 2 * W]``.

        Example:
            >>> import torch
            >>> block = RepUpBlock(
            ...     32, in_channels_next=16, out_channels=8, n_repeats=1
            ... )
            >>> x0, x1 = torch.zeros(1, 32, 4, 4), torch.zeros(1, 16, 8, 8)
            >>> conv_out, out = block(x0, x1)
            >>> conv_out.shape, out.shape
            (torch.Size([1, 8, 4, 4]), torch.Size([1, 8, 8, 8]))

        """
        conv_out = self.conv(x0)
        upsample_out = self.upsample(conv_out)
        concat_out = torch.cat([upsample_out, x1], dim=1)
        out = self.encode_block(concat_out)
        return conv_out, out


class RepUpBlock(PANUpBlockBase):
    """Top-down fusion step of `RepPANNeck` with RepVGG-style blocks.

    The encode block is a `BlockRepeater` of ``n_repeats``
    `GeneralReparameterizableBlock` layers. The first layer maps the
    concatenation to ``out_channels``. The other layers keep that
    number. `RepPANNeck` uses this step when ``block`` is
    ``"RepBlock"``, as in the ``"n"`` and ``"s"`` variants.

    Example:
        >>> import torch
        >>> block = RepUpBlock(
        ...     32, in_channels_next=16, out_channels=8, n_repeats=3
        ... )
        >>> block.encode_block(torch.zeros(1, 24, 4, 4)).shape
        torch.Size([1, 8, 4, 4])

    """

    def __init__(
        self,
        in_channels: int,
        in_channels_next: int,
        out_channels: int,
        n_repeats: int,
    ):
        """Initialize the upsampling layers and the RepVGG-style stack.

        Args:
            in_channels (int): Number of channels of the coarse input.
            in_channels_next (int): Number of channels of the finer
                input, which the step concatenates.
            out_channels (int): Number of output channels.
            n_repeats (int): Number of `GeneralReparameterizableBlock`
                layers. A value below ``1`` still builds one layer.

        """
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            encode_block=BlockRepeater(
                GeneralReparameterizableBlock,
                in_channels=in_channels_next + out_channels,
                out_channels=out_channels,
                n_repeats=n_repeats,
            ),
        )


class CSPUpBlock(PANUpBlockBase):
    """Top-down fusion step of `RepPANNeck` with a CSP block.

    The encode block is a `CSPStackRepBlock` that maps the
    concatenation to ``out_channels``. `RepPANNeck` uses this step when
    ``block`` is ``"CSPStackRepBlock"``, as in the ``"m"`` and ``"l"``
    variants.

    Example:
        >>> block = CSPUpBlock(
        ...     32, in_channels_next=16, out_channels=8, n_repeats=4, e=0.5
        ... )
        >>> block.encode_block.conv_1.out_channels
        4
        >>> len(block.encode_block.rep_stack)
        2

    """

    def __init__(
        self,
        in_channels: int,
        in_channels_next: int,
        out_channels: int,
        n_repeats: int,
        e: float,
    ):
        """Initialize the upsampling layers and the CSP block.

        Args:
            in_channels (int): Number of channels of the coarse input.
            in_channels_next (int): Number of channels of the finer
                input, which the step concatenates.
            out_channels (int): Number of output channels.
            n_repeats (int): Number of RepVGG-style blocks in the
                `CSPStackRepBlock`. Each `BottleRep` holds two of them.
                The stack has ``n_repeats // 2`` `BottleRep` blocks, with
                a minimum of one.
            e (float): Fraction of ``out_channels`` in each of the two
                paths of the `CSPStackRepBlock`.

        """
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            encode_block=CSPStackRepBlock(
                in_channels=in_channels_next + out_channels,
                out_channels=out_channels,
                n_blocks=n_repeats,
                e=e,
            ),
        )


class PANDownBlockBase(ABC, nn.Module):
    """Base class of the bottom-up fusion steps of `RepPANNeck`.

    A ``3x3`` `ConvBlock` with stride ``2``, padding ``1``, batch norm,
    and ReLU halves the height and the width of the fine input. An odd
    size rounds up. The block concatenates the result with the lateral
    input along the channel axis and runs ``encode_block`` on it. A
    subclass selects ``encode_block``.

    """

    def __init__(
        self,
        in_channels: int,
        downsample_out_channels: int,
        encode_block: nn.Module,
    ):
        """Initialize the downsampling convolution.

        Args:
            in_channels (int): Number of channels of the fine input.
            downsample_out_channels (int): Number of channels after the
                downsampling convolution.
            encode_block (``nn.Module``): Block that runs on the
                concatenation. Its input has ``downsample_out_channels``
                plus the channels of the lateral input.

        """
        super().__init__()

        self.downsample = ConvBlock(
            in_channels=in_channels,
            out_channels=downsample_out_channels,
            kernel_size=3,
            stride=2,
            padding=3 // 2,
        )
        self.encode_block = encode_block

    def forward(self, x0: Tensor, x1: Tensor) -> Tensor:
        """Downsample the fine map and fuse it with the lateral map.

        Args:
            x0 (``Tensor``): Fine map of shape ``[B, in_channels, H, W]``.
            x1 (``Tensor``): Lateral map of shape ``[B, C1, H / 2, W / 2]``.
                An odd ``H`` or ``W`` rounds up. In the subclasses, ``C1``
                is ``in_channels_next``.

        Returns:
            ``Tensor``: The output of ``encode_block``. In the subclasses,
            it has the shape ``[B, out_channels, H / 2, W / 2]``.

        Example:
            >>> import torch
            >>> block = RepDownBlock(
            ...     4, 8, in_channels_next=12, out_channels=16, n_repeats=1
            ... )
            >>> x0, x1 = torch.zeros(1, 4, 8, 8), torch.zeros(1, 12, 4, 4)
            >>> block(x0, x1).shape
            torch.Size([1, 16, 4, 4])

        """
        x = self.downsample(x0)
        x = torch.cat([x, x1], dim=1)
        return self.encode_block(x)


class RepDownBlock(PANDownBlockBase):
    """Bottom-up fusion step of `RepPANNeck` with RepVGG-style blocks.

    The encode block is a `BlockRepeater` of ``n_repeats``
    `GeneralReparameterizableBlock` layers. The first layer maps the
    concatenation to ``out_channels``. The other layers keep that
    number. `RepPANNeck` uses this step when ``block`` is
    ``"RepBlock"``, as in the ``"n"`` and ``"s"`` variants.

    Example:
        >>> import torch
        >>> block = RepDownBlock(
        ...     4, 8, in_channels_next=12, out_channels=16, n_repeats=3
        ... )
        >>> block.encode_block(torch.zeros(1, 20, 4, 4)).shape
        torch.Size([1, 16, 4, 4])

    """

    def __init__(
        self,
        in_channels: int,
        downsample_out_channels: int,
        in_channels_next: int,
        out_channels: int,
        n_repeats: int,
    ):
        """Initialize the downsampling layer and the RepVGG-style stack.

        Args:
            in_channels (int): Number of channels of the fine input.
            downsample_out_channels (int): Number of channels after the
                downsampling convolution.
            in_channels_next (int): Number of channels of the lateral
                input, which the step concatenates.
            out_channels (int): Number of output channels.
            n_repeats (int): Number of `GeneralReparameterizableBlock`
                layers. A value below ``1`` still builds one layer.

        """
        super().__init__(
            in_channels=in_channels,
            downsample_out_channels=downsample_out_channels,
            encode_block=BlockRepeater(
                GeneralReparameterizableBlock,
                n_repeats=n_repeats,
                in_channels=downsample_out_channels + in_channels_next,
                out_channels=out_channels,
            ),
        )


class CSPDownBlock(PANDownBlockBase):
    """Bottom-up fusion step of `RepPANNeck` with a CSP block.

    The encode block is a `CSPStackRepBlock` that maps the
    concatenation to ``out_channels``. `RepPANNeck` uses this step when
    ``block`` is ``"CSPStackRepBlock"``, as in the ``"m"`` and ``"l"``
    variants.

    Example:
        >>> import torch
        >>> block = CSPDownBlock(
        ...     4, 8, in_channels_next=12, out_channels=16, n_repeats=2, e=0.25
        ... )
        >>> block.encode_block.conv_1.out_channels
        4
        >>> x0, x1 = torch.zeros(1, 4, 8, 8), torch.zeros(1, 12, 4, 4)
        >>> block(x0, x1).shape
        torch.Size([1, 16, 4, 4])

    """

    def __init__(
        self,
        in_channels: int,
        downsample_out_channels: int,
        in_channels_next: int,
        out_channels: int,
        n_repeats: int,
        e: float,
    ):
        """Initialize the downsampling layer and the CSP block.

        Args:
            in_channels (int): Number of channels of the fine input.
            downsample_out_channels (int): Number of channels after the
                downsampling convolution.
            in_channels_next (int): Number of channels of the lateral
                input, which the step concatenates.
            out_channels (int): Number of output channels.
            n_repeats (int): Number of RepVGG-style blocks in the
                `CSPStackRepBlock`. Each `BottleRep` holds two of them.
                The stack has ``n_repeats // 2`` `BottleRep` blocks, with
                a minimum of one.
            e (float): Fraction of ``out_channels`` in each of the two
                paths of the `CSPStackRepBlock`.

        """
        super().__init__(
            in_channels=in_channels,
            downsample_out_channels=downsample_out_channels,
            encode_block=CSPStackRepBlock(
                in_channels=downsample_out_channels + in_channels_next,
                out_channels=out_channels,
                n_blocks=n_repeats,
                e=e,
            ),
        )
