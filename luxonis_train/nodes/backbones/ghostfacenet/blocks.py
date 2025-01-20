import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from luxonis_train.nodes.backbones.micronet.blocks import _make_divisible
from luxonis_train.nodes.blocks import SqueezeExciteBlock


class ModifiedGDC(nn.Module):
    def __init__(self, image_size, in_chs, num_classes, dropout, emb=512):
        super().__init__()

        if image_size % 32 == 0:
            self.conv_dw = nn.Conv2d(
                in_chs,
                in_chs,
                kernel_size=(image_size // 32),
                groups=in_chs,
                bias=False,
            )
        else:
            self.conv_dw = nn.Conv2d(
                in_chs,
                in_chs,
                kernel_size=(image_size // 32 + 1),
                groups=in_chs,
                bias=False,
            )
        self.bn1 = nn.BatchNorm2d(in_chs)
        self.dropout = nn.Dropout(dropout)

        self.conv = nn.Conv2d(in_chs, emb, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(emb)
        self.linear = (
            nn.Linear(emb, num_classes) if num_classes else nn.Identity()
        )

    def forward(self, inps):
        x = inps
        x = self.conv_dw(x)
        x = self.bn1(x)
        x = self.dropout(x)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.bn2(x)
        x = self.linear(x)
        return x


class GhostModuleV2(nn.Module):
    def __init__(
        self,
        inp,
        oup,
        kernel_size=1,
        ratio=2,
        dw_size=3,
        stride=1,
        prelu=True,
        mode=None,
        args=None,
    ):
        super(GhostModuleV2, self).__init__()
        self.mode = mode
        self.gate_fn = nn.Sigmoid()

        if self.mode in ["original"]:
            self.oup = oup
            init_channels = math.ceil(oup / ratio)
            new_channels = init_channels * (ratio - 1)
            self.primary_conv = nn.Sequential(
                nn.Conv2d(
                    inp,
                    init_channels,
                    kernel_size,
                    stride,
                    kernel_size // 2,
                    bias=False,
                ),
                nn.BatchNorm2d(init_channels),
                nn.PReLU() if prelu else nn.Sequential(),
            )
            self.cheap_operation = nn.Sequential(
                nn.Conv2d(
                    init_channels,
                    new_channels,
                    dw_size,
                    1,
                    dw_size // 2,
                    groups=init_channels,
                    bias=False,
                ),
                nn.BatchNorm2d(new_channels),
                nn.PReLU() if prelu else nn.Sequential(),
            )
        elif self.mode in ["attn"]:  # DFC
            self.oup = oup
            init_channels = math.ceil(oup / ratio)
            new_channels = init_channels * (ratio - 1)
            self.primary_conv = nn.Sequential(
                nn.Conv2d(
                    inp,
                    init_channels,
                    kernel_size,
                    stride,
                    kernel_size // 2,
                    bias=False,
                ),
                nn.BatchNorm2d(init_channels),
                nn.PReLU() if prelu else nn.Sequential(),
            )
            self.cheap_operation = nn.Sequential(
                nn.Conv2d(
                    init_channels,
                    new_channels,
                    dw_size,
                    1,
                    dw_size // 2,
                    groups=init_channels,
                    bias=False,
                ),
                nn.BatchNorm2d(new_channels),
                nn.PReLU() if prelu else nn.Sequential(),
            )
            self.short_conv = nn.Sequential(
                nn.Conv2d(
                    inp, oup, kernel_size, stride, kernel_size // 2, bias=False
                ),
                nn.BatchNorm2d(oup),
                nn.Conv2d(
                    oup,
                    oup,
                    kernel_size=(1, 5),
                    stride=1,
                    padding=(0, 2),
                    groups=oup,
                    bias=False,
                ),
                nn.BatchNorm2d(oup),
                nn.Conv2d(
                    oup,
                    oup,
                    kernel_size=(5, 1),
                    stride=1,
                    padding=(2, 0),
                    groups=oup,
                    bias=False,
                ),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.mode in ["original"]:
            x1 = self.primary_conv(x)
            x2 = self.cheap_operation(x1)
            out = torch.cat([x1, x2], dim=1)
            return out[:, : self.oup, :, :]
        elif self.mode in ["attn"]:
            res = self.short_conv(F.avg_pool2d(x, kernel_size=2, stride=2))
            x1 = self.primary_conv(x)
            x2 = self.cheap_operation(x1)
            out = torch.cat([x1, x2], dim=1)
            return out[:, : self.oup, :, :] * F.interpolate(
                self.gate_fn(res),
                size=(out.shape[-2], out.shape[-1]),
                mode="nearest",
            )


class GhostBottleneckV2(nn.Module):
    def __init__(
        self,
        in_chs,
        mid_chs,
        out_chs,
        dw_kernel_size=3,
        stride=1,
        act_layer=nn.PReLU,
        se_ratio=0.0,
        layer_id=None,
        args=None,
    ):
        super(GhostBottleneckV2, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0.0
        self.stride = stride

        assert layer_id is not None, "Layer ID must be explicitly provided"

        # Point-wise expansion
        if layer_id <= 1:
            self.ghost1 = GhostModuleV2(
                in_chs, mid_chs, prelu=True, mode="original", args=args
            )
        else:
            self.ghost1 = GhostModuleV2(
                in_chs, mid_chs, prelu=True, mode="attn", args=args
            )

        # Depth-wise convolution
        if self.stride > 1:
            self.conv_dw = nn.Conv2d(
                mid_chs,
                mid_chs,
                dw_kernel_size,
                stride=stride,
                padding=(dw_kernel_size - 1) // 2,
                groups=mid_chs,
                bias=False,
            )
            self.bn_dw = nn.BatchNorm2d(mid_chs)

        # Squeeze-and-excitation
        if has_se:
            reduced_chs = _make_divisible(mid_chs * se_ratio, 4)
            self.se = SqueezeExciteBlock(
                mid_chs, reduced_chs, True, activation=nn.PReLU()
            )
        else:
            self.se = None

        self.ghost2 = GhostModuleV2(
            mid_chs, out_chs, prelu=False, mode="original", args=args
        )

        # shortcut
        if in_chs == out_chs and self.stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_chs,
                    in_chs,
                    dw_kernel_size,
                    stride=stride,
                    padding=(dw_kernel_size - 1) // 2,
                    groups=in_chs,
                    bias=False,
                ),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_chs),
            )

    def forward(self, x):
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
