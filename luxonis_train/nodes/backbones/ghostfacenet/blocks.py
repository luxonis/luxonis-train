import math
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from luxonis_train.nodes.backbones.micronet.blocks import _make_divisible
from luxonis_train.nodes.blocks import SqueezeExciteBlock
from luxonis_train.nodes.blocks.blocks import ConvBlock


class OriginalGhostModuleV2(nn.Module):
    """Half the features from a conv, half from a cheap depthwise one."""

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
        super().__init__()
        self.out_channels = out_channels
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
        r"""Generate intrinsic and inexpensive GhostNet features.

        Args:
            x: Feature map used to generate intrinsic and ghost features.

        Returns:
            Concatenated intrinsic and ghost features.

        .. shape-contract::

            Inputs
                :math:`x`
                    :math:`(B, C_{\mathrm{in}}, H, W)`

            Outputs
                :math:`\mathrm{output}`
                    :math:`(B, C_{\mathrm{out}}, H_{\mathrm{out}}, W_{\mathrm{out}})`

        """  # noqa: E501
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, : self.out_channels, ...]


class AttentionGhostModuleV2(OriginalGhostModuleV2):
    """`OriginalGhostModuleV2` gated by a decoupled attention branch."""

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
        r"""Generate GhostNet features with spatial attention.

        Args:
            x: Feature map used to generate intrinsic, ghost, and
                attention features.

        Returns:
            Attention-gated GhostNet feature map.

        .. shape-contract::

            Inputs
                :math:`x`
                    :math:`(B, C_{\mathrm{in}}, H, W)`

            Outputs
                :math:`\mathrm{output}`
                    :math:`(B, C_{\mathrm{out}}, H_{\mathrm{out}}, W_{\mathrm{out}})`

        """  # noqa: E501
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)

        return out[:, : self.out_channels, ...] * F.interpolate(
            self.short_conv(x),
            size=(out.shape[-2], out.shape[-1]),
            mode="nearest",
        )


class GhostBottleneckV2(nn.Module):
    """Residual bottleneck built from two ghost modules."""

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
        super().__init__()
        has_se = se_ratio is not None and se_ratio > 0.0
        self.stride = stride

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
        if self.stride > 1:
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
        if in_channels == out_channels and self.stride == 1:
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
        r"""Apply a GhostNetV2 bottleneck.

        Args:
            x: Feature map supplied to the GhostNetV2 bottleneck.

        Returns:
            Feature map produced by the GhostNetV2 bottleneck.

        .. shape-contract::

            Inputs
                :math:`x`
                    :math:`(B, C_{\mathrm{in}}, H, W)`

            Outputs
                :math:`\mathrm{output}`
                    :math:`(B, C_{\mathrm{out}}, H_{\mathrm{out}}, W_{\mathrm{out}})`

        """  # noqa: E501
        residual = x
        x = self.ghost1(x)
        if self.stride > 1:
            x = self.conv_dw(x)
            x = self.bn_dw(x)
        if self.se is not None:
            x = self.se(x)
        x = self.ghost2(x)
        x += self.shortcut(residual)
        return x


class GhostBottleneckLayer(nn.Sequential):
    """One GhostFaceNet stage, a run of `GhostBottleneckV2` blocks."""

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

    def forward(self, x: Tensor) -> Tensor:
        r"""Apply a residual GhostNet bottleneck layer.

        Args:
            x: Feature map supplied to the GhostNet layer.

        Returns:
            Feature map produced by the residual GhostNet layer.

        .. shape-contract::

            Inputs
                :math:`x`
                    :math:`(B, C_{\mathrm{in}}, H, W)`

            Outputs
                :math:`\mathrm{output}`
                    :math:`(B, C_{\mathrm{out}}, H_{\mathrm{out}}, W_{\mathrm{out}})`

        """  # noqa: E501
        return super().forward(x)
