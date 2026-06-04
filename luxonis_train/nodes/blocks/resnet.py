from torch import Tensor, nn
from typeguard import typechecked

from .blocks import ConvBlock, DropPath


class GenericResidualBlock(nn.Module):
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
        out = self.block(x)
        out += self.shortcut(x)
        return self.final_relu(out)


class ResNetBlock(GenericResidualBlock):
    """A basic residual block for ResNet."""

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
        """Initialize a basic ResNet residual block.

        Args:
            in_channels (int): Number of input channels.
            hidden_channels (int): Number of output channels.
            stride (int): Stride for the convolutional layers. Defaults to 1.
            expansion (int): Expansion factor for the output channels. Defaults to 1.
            final_relu (bool): Whether to apply a ReLU activation after the residual addition. Defaults to True.
            droppath_prob (float): Drop path probability for stochastic depth. Defaults to 0.0.

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
    """A bottleneck block for ResNet."""

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
        """Initialize a ResNet bottleneck residual block.

        Args:
            in_channels (int): Number of input channels.
            hidden_channels (int): Number of intermediate channels.
            stride (int): Stride for the second convolutional layer. Defaults to 1.
            expansion (int): Expansion factor for the output channels. Defaults to 4.
            final_relu (bool): Whether to apply a ReLU activation after the residual addition. Defaults to True.
            droppath_prob (float): Drop path probability for stochastic depth. Defaults to 0.0.

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
