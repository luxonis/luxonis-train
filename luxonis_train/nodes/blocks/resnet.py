"""The residual and bottleneck blocks of ResNet."""

from torch import Tensor, nn
from typeguard import typechecked

from .blocks import ConvBlock, DropPath


class GenericResidualBlock(nn.Module):
    """Residual block that adds a shortcut to the output of any block.

    The shortcut is `torch.nn.Identity` when ``stride`` is ``1`` and
    ``in_channels`` equals ``expansion * hidden_channels``. Otherwise
    it is a ``1x1`` `ConvBlock` with ``stride``, a batch norm, and no
    activation. This projection maps the input to
    ``expansion * hidden_channels`` channels. `ResNetBlock` and
    `ResNetBottleneck` build on this class.

    Attributes:
        block (``nn.Module``): The residual branch.
        shortcut (``nn.Module``): The identity or the ``1x1``
            projection.
        final_relu (``nn.Module``): `torch.nn.ReLU` or
            `torch.nn.Identity`.

    """

    @typechecked
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        stride: int,
        expansion: int,
        final_relu: bool,
        block: nn.Module,
    ):
        """Store the branch and build the shortcut and the activation.

        Args:
            in_channels (int): The number of input channels.
            hidden_channels (int): The base width. The output has
                ``expansion * hidden_channels`` channels.
            stride (int): The stride of the shortcut projection. The
                branch must reduce the size by the same factor.
            expansion (int): The factor from ``hidden_channels`` to the
                number of output channels.
            final_relu (bool): Whether a ReLU follows the sum.
            block (``nn.Module``): The residual branch. Its output must
                have the shape of the shortcut output.

        """
        super().__init__()
        self.block = block

        if stride != 1 or in_channels != expansion * hidden_channels:
            self.shortcut = ConvBlock(
                in_channels,
                expansion * hidden_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
                activation=None,
            )
        else:
            self.shortcut = nn.Identity()
        if final_relu:
            self.final_relu = nn.ReLU()
        else:
            self.final_relu = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """Add the shortcut to the output of the branch.

        The addition runs in place on the output of ``block``.

        Args:
            x (``Tensor``): The input of shape ``[B, in_channels, H, W]``.

        Returns:
            ``Tensor``: The sum ``block(x) + shortcut(x)``, after the ReLU
            when the constructor got ``final_relu=True``. The shape is
            ``[B, expansion * hidden_channels, H', W']``, where ``stride``
            sets ``H'`` and ``W'``.

        """
        out = self.block(x)
        out += self.shortcut(x)
        return self.final_relu(out)


class ResNetBlock(GenericResidualBlock):
    """Basic ResNet block with two ``3x3`` convolutions.

    The residual branch is a ``3x3`` convolution with ``stride``, a batch
    norm, a ReLU, a ``3x3`` convolution, a batch norm, and `DropPath`.
    The convolutions have no bias. The shortcut follows the rules of
    `GenericResidualBlock`.

    Example:
        >>> import torch
        >>> from torch import nn
        >>> from luxonis_train.nodes.blocks import ResNetBlock
        >>> isinstance(ResNetBlock(8, 8).shortcut, nn.Identity)
        True
        >>> block = ResNetBlock(8, 16, stride=2)
        >>> block(torch.zeros(1, 8, 8, 8)).shape
        torch.Size([1, 16, 4, 4])

    """

    @typechecked
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        stride: int = 1,
        expansion: int = 1,
        final_relu: bool = True,
        droppath_prob: float = 0.0,
    ):
        """Build the residual branch of the block.

        Args:
            in_channels (int): The number of input channels.
            hidden_channels (int): The number of output channels.
            stride (int): The stride of the first convolution and of the
                shortcut.
            expansion (int): The factor of the shortcut channels. The
                branch always gives ``hidden_channels`` channels, so
                only ``1`` works. With a value above ``1``, ``forward``
                raises ``RuntimeError``.
            final_relu (bool): Whether a ReLU follows the sum.
            droppath_prob (float): The drop probability of the `DropPath`
                at the end of the branch.

        """
        super().__init__(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            stride=stride,
            expansion=expansion,
            final_relu=final_relu,
            block=nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    hidden_channels,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU(),
                nn.Conv2d(
                    hidden_channels,
                    hidden_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(hidden_channels),
                DropPath(drop_prob=droppath_prob),
            ),
        )


class ResNetBottleneck(GenericResidualBlock):
    """ResNet bottleneck block of three convolutions.

    The residual branch reduces the input to ``hidden_channels`` with a
    ``1x1`` convolution. A ``3x3`` convolution with ``stride`` follows.
    A last ``1x1`` convolution expands the result to
    ``expansion * hidden_channels`` channels. Each convolution has a
    batch norm and no bias. ReLUs follow the first two batch norms, and
    `DropPath` ends the branch. The shortcut follows the rules of
    `GenericResidualBlock`.

    Example:
        >>> import torch
        >>> from luxonis_train.nodes.blocks import ResNetBottleneck
        >>> block = ResNetBottleneck(16, 8, stride=2)
        >>> block(torch.zeros(1, 16, 8, 8)).shape
        torch.Size([1, 32, 4, 4])

    """

    @typechecked
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        stride: int = 1,
        expansion: int = 4,
        final_relu: bool = True,
        droppath_prob: float = 0.0,
    ):
        """Build the residual branch of the block.

        Args:
            in_channels (int): The number of input channels.
            hidden_channels (int): The number of channels of the ``1x1``
                reduction and of the ``3x3`` convolution.
            stride (int): The stride of the ``3x3`` convolution and of
                the shortcut.
            expansion (int): The output has ``expansion * hidden_channels``
                channels.
            final_relu (bool): Whether a ReLU follows the sum.
            droppath_prob (float): The drop probability of the `DropPath`
                at the end of the branch.

        """
        super().__init__(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            stride=stride,
            expansion=expansion,
            final_relu=final_relu,
            block=nn.Sequential(
                nn.Conv2d(
                    in_channels, hidden_channels, kernel_size=1, bias=False
                ),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU(),
                nn.Conv2d(
                    hidden_channels,
                    hidden_channels,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU(),
                nn.Conv2d(
                    hidden_channels,
                    expansion * hidden_channels,
                    kernel_size=1,
                    bias=False,
                ),
                nn.BatchNorm2d(expansion * hidden_channels),
                DropPath(drop_prob=droppath_prob),
            ),
        )
