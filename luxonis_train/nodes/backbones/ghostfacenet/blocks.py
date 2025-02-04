import math
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from luxonis_train.nodes.backbones.micronet.blocks import _make_divisible
from luxonis_train.nodes.blocks import SqueezeExciteBlock
from luxonis_train.nodes.blocks.blocks import ConvModule


class GhostModuleV2(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mode: Literal["original", "attn"],
        kernel_size: int = 1,
        ratio: int = 2,
        dw_size: int = 3,
        stride: int = 1,
        use_prelu: bool = True,
    ):
        super().__init__()
        self.mode = mode
        self.out_channels = out_channels
        intermediate_channels = math.ceil(out_channels / ratio)
        new_channels = intermediate_channels * (ratio - 1)
        self.primary_conv = ConvModule(
            in_channels,
            intermediate_channels,
            kernel_size,
            stride,
            kernel_size // 2,
            activation=nn.PReLU() if use_prelu else False,
        )
        self.cheap_operation = ConvModule(
            intermediate_channels,
            new_channels,
            dw_size,
            1,
            dw_size // 2,
            groups=intermediate_channels,
            activation=nn.PReLU() if use_prelu else False,
        )

        if self.mode == "attn":
            self.short_conv = nn.Sequential(
                ConvModule(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    kernel_size // 2,
                    activation=False,
                ),
                ConvModule(
                    out_channels,
                    out_channels,
                    kernel_size=(1, 5),
                    stride=1,
                    padding=(0, 2),
                    groups=out_channels,
                    activation=False,
                ),
                ConvModule(
                    out_channels,
                    out_channels,
                    kernel_size=(5, 1),
                    stride=1,
                    padding=(2, 0),
                    groups=out_channels,
                    activation=False,
                ),
                nn.AvgPool2d(kernel_size=2, stride=2),
                nn.Sigmoid(),
            )

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        if self.mode == "original":
            return out[:, : self.out_channels, ...]

        return out[:, : self.out_channels, ...] * F.interpolate(
            self.short_conv(x),
            size=(out.shape[-2], out.shape[-1]),
            mode="nearest",
        )


class GhostBottleneckV2(nn.Module):
    def __init__(
        self,
        in_channels: int,
        intermediate_channels: int,
        out_channels: int,
        dw_kernel_size: int = 3,
        stride: int = 1,
        se_ratio: float = 0.0,
        *,
        layer_id: int,
    ):
        super().__init__()
        has_se = se_ratio is not None and se_ratio > 0.0
        self.stride = stride

        # Point-wise expansion
        if layer_id <= 1:
            self.ghost1 = GhostModuleV2(
                in_channels,
                intermediate_channels,
                use_prelu=True,
                mode="original",
            )
        else:
            self.ghost1 = GhostModuleV2(
                in_channels, intermediate_channels, use_prelu=True, mode="attn"
            )

        # Depth-wise convolution
        if self.stride > 1:
            self.conv_dw = nn.Conv2d(
                intermediate_channels,
                intermediate_channels,
                dw_kernel_size,
                stride=stride,
                padding=(dw_kernel_size - 1) // 2,
                groups=intermediate_channels,
                bias=False,
            )
            self.bn_dw = nn.BatchNorm2d(intermediate_channels)

        # Squeeze-and-excitation
        if has_se:
            reduced_chs = _make_divisible(
                int(intermediate_channels * se_ratio), 4
            )
            self.se = SqueezeExciteBlock(
                intermediate_channels, reduced_chs, True, activation=nn.PReLU()
            )
        else:
            self.se = None

        self.ghost2 = GhostModuleV2(
            intermediate_channels,
            out_channels,
            use_prelu=False,
            mode="original",
        )

        # shortcut
        if in_channels == out_channels and self.stride == 1:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    dw_kernel_size,
                    stride=stride,
                    padding=(dw_kernel_size - 1) // 2,
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
