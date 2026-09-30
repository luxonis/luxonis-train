"""The ghost modules of GhostFaceNet, which produce part of the feature
maps with cheap operations instead of full convolutions.
"""

import math
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from luxonis_train.nodes.backbones.micronet.blocks import _make_divisible
from luxonis_train.nodes.blocks import SqueezeExciteBlock
from luxonis_train.nodes.blocks.blocks import ConvBlock


class OriginalGhostModuleV2(nn.Module):
    """Ghost module that makes part of its output with a cheap operation.

    A primary convolution makes ``ceil(out_channels / ratio)`` channels.
    A depthwise convolution, the cheap operation, makes ``ratio - 1``
    channels from each of them. The module concatenates both results
    and keeps the first ``out_channels`` channels. Both convolutions
    have a batch norm.

    Example:
        >>> import torch
        >>> module = OriginalGhostModuleV2(4, 7, ratio=3)
        >>> module.primary_conv.out_channels
        3
        >>> module.cheap_operation.out_channels
        6
        >>> module(torch.zeros(1, 4, 8, 8)).shape
        torch.Size([1, 7, 8, 8])

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        ratio: int = 2,
        dw_size: int = 3,
        stride: int = 1,
        use_prelu: bool = True,
    ):
        """Build the primary and the cheap convolutions.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Size of the primary kernel. The padding is
                ``kernel_size // 2``.
            ratio (int): The ratio of ``out_channels`` to the channels of
                the primary convolution.
            dw_size (int): Size of the depthwise kernel. The padding is
                ``dw_size // 2``.
            stride (int): Stride of the primary convolution.
            use_prelu (bool): Whether both convolutions end with
                `torch.nn.PReLU`. When ``False``, they have no
                activation.

        """
        super().__init__()
        self._out_channels = out_channels
        intermediate_channels = math.ceil(out_channels / ratio)
        new_channels = intermediate_channels * (ratio - 1)
        self.primary_conv = ConvBlock(
            in_channels,
            intermediate_channels,
            kernel_size,
            stride,
            kernel_size // 2,
            activation=nn.PReLU() if use_prelu else None,
        )
        self.cheap_operation = ConvBlock(
            intermediate_channels,
            new_channels,
            dw_size,
            1,
            dw_size // 2,
            groups=intermediate_channels,
            activation=nn.PReLU() if use_prelu else None,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Concatenate the primary and the cheap features.

        Args:
            x (``Tensor``): Input of shape ``[B, in_channels, H, W]``.

        Returns:
            ``Tensor``: Output of shape ``[B, out_channels, H', W']``. The
            stride of the primary convolution sets ``H'`` and ``W'``.

        """
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, : self._out_channels, ...]


class AttentionGhostModuleV2(OriginalGhostModuleV2):
    """Ghost module with a gate from a decoupled attention branch.

    The attention branch reads the input of the module. It runs a
    convolution with the kernel size and the stride of the primary
    convolution, a ``1x5`` and a ``5x1`` depthwise convolution, a
    ``2x2`` average pool with stride 2, and a sigmoid. The convolutions
    of the branch have a batch norm and no activation. GhostNetV2 calls
    this branch DFC attention. A nearest interpolation resizes the gate
    to the output size, and the module multiplies the ghost features by
    the gate.

    Example:
        >>> import torch
        >>> module = AttentionGhostModuleV2(4, 8)
        >>> module(torch.zeros(1, 4, 7, 7)).shape
        torch.Size([1, 8, 7, 7])

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        ratio: int = 2,
        dw_size: int = 3,
        stride: int = 1,
        use_prelu: bool = True,
    ):
        """Build the ghost convolutions and the attention branch.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Size of the primary kernel and of the
                first kernel of the attention branch. The padding is
                ``kernel_size // 2``.
            ratio (int): The ratio of ``out_channels`` to the channels of
                the primary convolution.
            dw_size (int): Size of the cheap depthwise kernel. The
                padding is ``dw_size // 2``.
            stride (int): Stride of the primary convolution and of the
                first convolution of the attention branch.
            use_prelu (bool): Whether the primary and the cheap
                convolutions end with `torch.nn.PReLU`. The attention
                branch never has an activation before the sigmoid.

        """
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            ratio,
            dw_size,
            stride,
            use_prelu,
        )

        self.short_conv = nn.Sequential(
            ConvBlock(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                kernel_size // 2,
                activation=None,
            ),
            ConvBlock(
                out_channels,
                out_channels,
                kernel_size=(1, 5),
                stride=1,
                padding=(0, 2),
                groups=out_channels,
                activation=None,
            ),
            ConvBlock(
                out_channels,
                out_channels,
                kernel_size=(5, 1),
                stride=1,
                padding=(2, 0),
                groups=out_channels,
                activation=None,
            ),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Multiply the ghost features by the attention gate.

        Args:
            x (``Tensor``): Input of shape ``[B, in_channels, H, W]``.

        Returns:
            ``Tensor``: Output of shape ``[B, out_channels, H', W']``. The
            stride of the primary convolution sets ``H'`` and ``W'``.
            Each value is the ghost feature times a gate between ``0``
            and ``1``.

        """
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)

        return out[:, : self._out_channels, ...] * F.interpolate(
            self.short_conv(x),
            size=(out.shape[-2], out.shape[-1]),
            mode="nearest",
        )


class GhostBottleneckV2(nn.Module):
    """Ghost bottleneck of GhostFaceNetsV2 with a shortcut.

    The main path has these layers:

    - A ghost module with `torch.nn.PReLU` expands ``in_channels`` to
      ``hidden_channels``.
    - For a ``stride`` above ``1``, a depthwise convolution with a batch
      norm and no activation reduces the spatial size.
    - For a ``se_ratio`` above ``0``, a `SqueezeExciteBlock` with a hard
      sigmoid and `torch.nn.PReLU` scales the channels.
    - An `OriginalGhostModuleV2` without an activation projects to
      ``out_channels``.

    The shortcut is `torch.nn.Identity` when ``in_channels`` equals
    ``out_channels`` and ``stride`` is ``1``. Otherwise it is a depthwise
    convolution with ``kernel_size`` and ``stride``, then a ``1x1``
    convolution, each with a batch norm. The block adds the shortcut to
    the main path.

    Example:
        >>> import torch
        >>> block = GhostBottleneckV2(
        ...     8, 16, 8, stride=2, se_ratio=0.25, mode="attention"
        ... )
        >>> block(torch.zeros(1, 8, 8, 8)).shape
        torch.Size([1, 8, 4, 4])
        >>> GhostBottleneckV2(8, 16, 8, mode="original").shortcut
        Identity()

    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        se_ratio: float = 0.0,
        *,
        mode: Literal["original", "attention"],
    ):
        """Build the main path and the shortcut.

        Args:
            in_channels (int): Number of input channels.
            hidden_channels (int): Number of channels after the
                expansion.
            out_channels (int): Number of output channels.
            kernel_size (int): Size of the depthwise kernels of the main
                path and of the shortcut. The padding is
                ``(kernel_size - 1) // 2``.
            stride (int): Stride of both depthwise convolutions.
            se_ratio (float): The ratio of the squeeze-and-excite
                channels to ``hidden_channels``. The block rounds the
                result to a multiple of 4. ``0`` or less adds no
                `SqueezeExciteBlock`.
            mode (``Literal["original", "attention"]``): The ghost module
                of the expansion. ``"original"`` selects
                `OriginalGhostModuleV2`, and ``"attention"`` selects
                `AttentionGhostModuleV2`.

        """
        super().__init__()
        has_se = se_ratio is not None and se_ratio > 0.0
        self._stride = stride

        # Point-wise expansion
        if mode == "original":
            self.ghost1 = OriginalGhostModuleV2(
                in_channels, hidden_channels, use_prelu=True
            )
        else:
            self.ghost1 = AttentionGhostModuleV2(
                in_channels, hidden_channels, use_prelu=True
            )

        # Depth-wise convolution
        if self._stride > 1:
            self.conv_dw = nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size,
                stride=stride,
                padding=(kernel_size - 1) // 2,
                groups=hidden_channels,
                bias=False,
            )
            self.bn_dw = nn.BatchNorm2d(hidden_channels)

        # Squeeze-and-excitation
        if has_se:
            reduced_chs = _make_divisible(int(hidden_channels * se_ratio), 4)
            self.se = SqueezeExciteBlock(
                hidden_channels,
                reduced_chs,
                hard_sigmoid=True,
                activation=nn.PReLU(),
            )
        else:
            self.se = None

        self.ghost2 = OriginalGhostModuleV2(
            hidden_channels, out_channels, use_prelu=False
        )

        # shortcut
        if in_channels == out_channels and self._stride == 1:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size,
                    stride=stride,
                    padding=(kernel_size - 1) // 2,
                    groups=in_channels,
                    bias=False,
                ),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: Tensor) -> Tensor:
        """Apply the main path and add the shortcut.

        Args:
            x (``Tensor``): Input of shape ``[B, in_channels, H, W]``.

        Returns:
            ``Tensor``: Output of shape ``[B, out_channels, H', W']``. For
            an odd ``kernel_size``, ``H'`` and ``W'`` are ``H`` and ``W``
            divided by ``stride`` and rounded up.

        """
        residual = x
        x = self.ghost1(x)
        if self._stride > 1:
            x = self.conv_dw(x)
            x = self.bn_dw(x)
        if self.se is not None:
            x = self.se(x)
        x = self.ghost2(x)
        x += self.shortcut(residual)
        return x


class GhostBottleneckLayer(nn.Sequential):
    """Stage of `GhostBottleneckV2` blocks with one ghost module mode.

    The five lists give one value for each block. Each block reads the
    output of the block before it. The layer multiplies the expansion
    and output channels by ``width_multiplier``. It then rounds each
    count to the nearest multiple of 4, with a minimum of 4. A count
    that rounds below 90% of its value goes up by 4.

    Attributes:
        output_channel (int): Number of output channels of the last
            block. It is ``input_channel`` when the lists are empty.

    Example:
        >>> import torch
        >>> layer = GhostBottleneckLayer(
        ...     width_multiplier=1,
        ...     input_channel=16,
        ...     kernel_sizes=[3, 3],
        ...     expand_sizes=[48, 72],
        ...     output_channels=[24, 24],
        ...     se_ratios=[0.0, 0.25],
        ...     strides=[2, 1],
        ...     mode="attention",
        ... )
        >>> len(layer), layer.output_channel
        (2, 24)
        >>> layer(torch.zeros(1, 16, 8, 8)).shape
        torch.Size([1, 24, 4, 4])

    """

    def __init__(
        self,
        width_multiplier: int,
        input_channel: int,
        kernel_sizes: list[int],
        expand_sizes: list[int],
        output_channels: list[int],
        se_ratios: list[float],
        strides: list[int],
        mode: Literal["original", "attention"],
    ):
        """Build one `GhostBottleneckV2` for each entry of the lists.

        Args:
            width_multiplier (int): The scale of ``expand_sizes`` and
                ``output_channels``.
            input_channel (int): Number of input channels of the first
                block. The layer does not scale it.
            kernel_sizes (list[int]): The ``kernel_size`` of each block.
            expand_sizes (list[int]): The ``hidden_channels`` of each
                block, before ``width_multiplier``.
            output_channels (list[int]): The ``out_channels`` of each
                block, before ``width_multiplier``.
            se_ratios (list[float]): The ``se_ratio`` of each block.
            strides (list[int]): The ``stride`` of each block.
            mode (``Literal["original", "attention"]``): The ``mode`` of
                all blocks.

        Raises:
            ValueError: When the five lists do not have the same length.

        """
        blocks = []
        for (
            kernel_size,
            expand_size,
            output_channel,
            se_ratio,
            stride,
        ) in zip(
            kernel_sizes,
            expand_sizes,
            output_channels,
            se_ratios,
            strides,
            strict=True,
        ):
            hidden_channel = _make_divisible(expand_size * width_multiplier, 4)
            output_channel = _make_divisible(
                output_channel * width_multiplier, 4
            )
            blocks.append(
                GhostBottleneckV2(
                    input_channel,
                    hidden_channel,
                    output_channel,
                    kernel_size=kernel_size,
                    stride=stride,
                    se_ratio=se_ratio,
                    mode=mode,
                )
            )
            input_channel = output_channel

        self.output_channel = input_channel

        super().__init__(*blocks)
