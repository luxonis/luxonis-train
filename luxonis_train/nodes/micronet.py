from typing import Literal

import torch
from torch import Tensor, nn

from luxonis_train.nodes.activations import HSigmoid, HSwish
from luxonis_train.nodes.blocks import ConvModule

from .base_node import BaseNode


class MicroNet(BaseNode[Tensor, list[Tensor]]):
    """

    TODO: DOCS
    """

    attach_index: int = -1

    def __init__(self, variant: Literal["M1", "M2", "M3"] = "M1", **kwargs):
        """MicroNet backbone.

        @type variant: Literal["M1", "M2", "M3"]
        @param variant: Model variant to use. Defaults to "M1".
        """
        super().__init__(**kwargs)

        if variant not in MICRONET_VARIANTS_SETTINGS:
            raise ValueError(
                f"MicroNet model variant should be in {list(MICRONET_VARIANTS_SETTINGS.keys())}"
            )

        self.inplanes = 64
        (
            in_channels,
            stem_groups,
            _,
            init_a,
            init_b,
            out_indices,
            channels,
            cfgs,
        ) = MICRONET_VARIANTS_SETTINGS[variant]
        self.out_indices = out_indices
        self.channels = channels

        self.features = nn.ModuleList([Stem(3, 2, stem_groups)])

        for (
            stride,
            out_channels,
            kernel_size,
            c1,
            c2,
            g1,
            g2,
            _,
            g3,
            g4,
            y1,
            y2,
            y3,
            r,
        ) in cfgs:
            self.features.append(
                MicroBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    (c1, c2),
                    (g1, g2),
                    (g3, g4),
                    (y1, y2, y3),
                    r,
                    init_a,
                    init_b,
                )
            )
            in_channels = out_channels

    def forward(self, x: Tensor) -> list[Tensor]:
        outs = []
        for m in self.features:
            x = m(x)
            outs.append(x)
        return outs


class MicroBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        t1: tuple[int, int] = (2, 2),
        gs1: tuple[int, int] = (0, 6),
        groups_1x1: tuple[int, int] = (1, 1),
        dy: tuple[int, int, int] = (2, 0, 1),
        r: int = 1,
        init_a: tuple[float, float] = (1.0, 1.0),
        init_b: tuple[float, float] = (0.0, 0.0),
    ):
        super().__init__()

        self.identity = stride == 1 and in_channels == out_channels
        y1, y2, y3 = dy
        g1, g2 = groups_1x1
        reduction = 8 * r
        intermediate_channels = in_channels * t1[0] * t1[1]

        if gs1[0] == 0:
            self.layers = nn.Sequential(
                DepthSpatialSepConv(in_channels, t1, kernel_size, stride),
                DYShiftMax(
                    intermediate_channels,
                    intermediate_channels,
                    init_a,
                    init_b,
                    True if y2 == 2 else False,
                    gs1[1],
                    reduction,
                )
                if y2 > 0
                else nn.ReLU6(True),
                ChannelShuffle(gs1[1]),
                ChannelShuffle(intermediate_channels // 2)
                if y2 != 0
                else nn.Sequential(),
                ConvModule(
                    in_channels=intermediate_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    groups=g1,
                    activation=nn.Identity(),
                ),
                DYShiftMax(
                    out_channels,
                    out_channels,
                    (1.0, 0.0),
                    (0.0, 0.0),
                    False,
                    g2,
                    reduction // 2,
                )
                if y3 > 0
                else nn.Sequential(),
                ChannelShuffle(g2),
                ChannelShuffle(out_channels // 2)
                if out_channels % 2 == 0 and y3 != 0
                else nn.Sequential(),
            )
        elif g2 == 0:
            self.layers = nn.Sequential(
                ConvModule(
                    in_channels=in_channels,
                    out_channels=intermediate_channels,
                    kernel_size=1,
                    groups=gs1[0],
                    activation=nn.Identity(),
                ),
                DYShiftMax(
                    intermediate_channels,
                    intermediate_channels,
                    (1.0, 0.0),
                    (0.0, 0.0),
                    False,
                    gs1[1],
                    reduction,
                )
                if y3 > 0
                else nn.Sequential(),
            )
        else:
            self.layers = nn.Sequential(
                ConvModule(
                    in_channels=in_channels,
                    out_channels=intermediate_channels,
                    kernel_size=1,
                    groups=gs1[0],
                    activation=nn.Identity(),
                ),
                DYShiftMax(
                    intermediate_channels,
                    intermediate_channels,
                    init_a,
                    init_b,
                    True if y1 == 2 else False,
                    gs1[1],
                    reduction,
                )
                if y1 > 0
                else nn.ReLU6(True),
                ChannelShuffle(gs1[1]),
                DepthSpatialSepConv(intermediate_channels, (1, 1), kernel_size, stride),
                nn.Sequential(),
                DYShiftMax(
                    intermediate_channels,
                    intermediate_channels,
                    init_a,
                    init_b,
                    True if y2 == 2 else False,
                    gs1[1],
                    reduction,
                    True,
                )
                if y2 > 0
                else nn.ReLU6(True),
                ChannelShuffle(intermediate_channels // 4)
                if y1 != 0 and y2 != 0
                else nn.Sequential()
                if y1 == 0 and y2 == 0
                else ChannelShuffle(intermediate_channels // 2),
                ConvModule(
                    in_channels=intermediate_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    groups=g1,
                    activation=nn.Identity(),
                ),
                DYShiftMax(
                    out_channels,
                    out_channels,
                    (1.0, 0.0),
                    (0.0, 0.0),
                    False,
                    g2,
                    reduction=reduction // 2
                    if out_channels < intermediate_channels
                    else reduction,
                )
                if y3 > 0
                else nn.Sequential(),
                ChannelShuffle(g2),
                ChannelShuffle(out_channels // 2) if y3 != 0 else nn.Sequential(),
            )

    def forward(self, x: Tensor):
        identity = x
        out = self.layers(x)
        if self.identity:
            out += identity
        return out


class ChannelShuffle(nn.Module):
    def __init__(self, groups: int):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        b, c, h, w = x.size()
        channels_per_group = c // self.groups
        # reshape
        x = x.view(b, self.groups, channels_per_group, h, w)
        x = torch.transpose(x, 1, 2).contiguous()
        out = x.view(b, -1, h, w)
        return out


class DYShiftMax(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        init_a: tuple[float, float] = (0.0, 0.0),
        init_b: tuple[float, float] = (0.0, 0.0),
        act_relu: bool = True,
        g: int = 6,
        reduction: int = 4,
        expansion: bool = False,
    ):
        super().__init__()
        self.exp: Literal[2, 4] = 4 if act_relu else 2
        self.init_a = init_a
        self.init_b = init_b
        self.out_channels = out_channels

        self.avg_pool = nn.Sequential(nn.Sequential(), nn.AdaptiveAvgPool2d(1))

        squeeze = self._make_divisible(in_channels // reduction, 4)

        self.fc = nn.Sequential(
            nn.Linear(in_channels, squeeze),
            nn.ReLU(True),
            nn.Linear(squeeze, out_channels * self.exp),
            HSigmoid(),
        )

        if g != 1 and expansion:
            g = in_channels // g

        gc = in_channels // g
        index = Tensor(range(in_channels)).view(1, in_channels, 1, 1)
        index = index.view(1, g, gc, 1, 1)
        indexgs = torch.split(index, [1, g - 1], dim=1)
        indexgs = torch.cat([indexgs[1], indexgs[0]], dim=1)
        indexs = torch.split(indexgs, [1, gc - 1], dim=2)
        indexs = torch.cat([indexs[1], indexs[0]], dim=2)
        self.index = indexs.view(in_channels).long()

    def forward(self, x: Tensor):
        B, C, _, _ = x.shape
        x_out = x

        y = self.avg_pool(x).view(B, C)
        y = self.fc(y).view(B, -1, 1, 1)
        y = (y - 0.5) * 4.0

        x2 = x_out[:, self.index, :, :]

        if self.exp == 4:
            a1, b1, a2, b2 = torch.split(y, self.out_channels, dim=1)

            a1 = a1 + self.init_a[0]
            a2 = a2 + self.init_b[1]
            b1 = b1 + self.init_b[0]
            b2 = b2 + self.init_b[1]

            z1 = x_out * a1 + x2 * b1
            z2 = x_out * a2 + x2 * b2

            out = torch.max(z1, z2)

        elif self.exp == 2:
            a1, b1 = torch.split(y, self.out_channels, dim=1)
            a1 = a1 + self.init_a[0]
            b1 = b1 + self.init_b[0]
            out = x_out * a1 + x2 * b1
        else:
            raise RuntimeError("Expansion should be 2 or 4.")

        return out

    def _make_divisible(self, v, divisor, min_value=None):
        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v


class SwishLinear(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(in_channels, out_channels), nn.BatchNorm1d(out_channels), HSwish()
        )

    def forward(self, x: Tensor):
        return self.linear(x)


class SpatialSepConvSF(nn.Module):
    def __init__(
        self, in_channels: int, outs: tuple[int, int], kernel_size: int, stride: int
    ):
        super().__init__()
        out_channels1, out_channels2 = outs
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels1,
                (kernel_size, 1),
                (stride, 1),
                (kernel_size // 2, 0),
                bias=False,
            ),
            nn.BatchNorm2d(out_channels1),
            nn.Conv2d(
                out_channels1,
                out_channels1 * out_channels2,
                (1, kernel_size),
                (1, stride),
                (0, kernel_size // 2),
                groups=out_channels1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels1 * out_channels2),
            ChannelShuffle(out_channels1),
        )

    def forward(self, x: Tensor):
        return self.conv(x)


class Stem(nn.Module):
    def __init__(self, in_channels: int, stride: int, outs: tuple[int, int] = (4, 4)):
        super().__init__()
        self.stem = nn.Sequential(
            SpatialSepConvSF(in_channels, outs, 3, stride), nn.ReLU6(True)
        )

    def forward(self, x: Tensor):
        return self.stem(x)


class DepthSpatialSepConv(nn.Module):
    def __init__(
        self, in_channels: int, expand: tuple[int, int], kernel_size: int, stride: int
    ):
        super().__init__()
        exp1, exp2 = expand
        intermediate_channels = in_channels * exp1
        out_channels = in_channels * exp1 * exp2

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                intermediate_channels,
                (kernel_size, 1),
                (stride, 1),
                (kernel_size // 2, 0),
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(intermediate_channels),
            nn.Conv2d(
                intermediate_channels,
                out_channels,
                (1, kernel_size),
                (1, stride),
                (0, kernel_size // 2),
                groups=intermediate_channels,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: Tensor):
        return self.conv(x)


MICRONET_VARIANTS_SETTINGS = {
    "M1": [
        6,  # stem_ch
        [3, 2],  # stem_groups
        960,  # out_ch
        [1.0, 1.0],  # init_a
        [0.0, 0.0],  # init_b
        [1, 2, 4, 7],  # out indices
        [8, 16, 32, 576],
        [
            # s, c, ks, c1, c2, g1, g2, c3, g3, g4, y1, y2, y3, r
            [2, 8, 3, 2, 2, 0, 6, 8, 2, 2, 2, 0, 1, 1],
            [2, 16, 3, 2, 2, 0, 8, 16, 4, 4, 2, 2, 1, 1],
            [
                2,
                16,
                5,
                2,
                2,
                0,
                16,
                16,
                4,
                4,
                2,
                2,
                1,
                1,
            ],
            [
                1,
                32,
                5,
                1,
                6,
                4,
                4,
                32,
                4,
                4,
                2,
                2,
                1,
                1,
            ],
            [
                2,
                64,
                5,
                1,
                6,
                8,
                8,
                64,
                8,
                8,
                2,
                2,
                1,
                1,
            ],
            [
                1,
                96,
                3,
                1,
                6,
                8,
                8,
                96,
                8,
                8,
                2,
                2,
                1,
                2,
            ],
            [1, 576, 3, 1, 6, 12, 12, 0, 0, 0, 2, 2, 1, 2],  # 96->96(4,24)->576
        ],
    ],
    "M2": [
        8,
        [4, 2],
        1024,
        [1.0, 1.0],
        [0.0, 0.0],
        [1, 3, 6, 9],
        [12, 24, 64, 768],
        [
            # s,  c, ks, c1, c2, g1, g2, c3, g3, g4, y1, y2, y3, r
            [
                2,
                12,
                3,
                2,
                2,
                0,
                8,
                12,
                4,
                4,
                2,
                0,
                1,
                1,
            ],
            [
                2,
                16,
                3,
                2,
                2,
                0,
                12,
                16,
                4,
                4,
                2,
                2,
                1,
                1,
            ],
            [
                1,
                24,
                3,
                2,
                2,
                0,
                16,
                24,
                4,
                4,
                2,
                2,
                1,
                1,
            ],
            [
                2,
                32,
                5,
                1,
                6,
                6,
                6,
                32,
                4,
                4,
                2,
                2,
                1,
                1,
            ],
            [
                1,
                32,
                5,
                1,
                6,
                8,
                8,
                32,
                4,
                4,
                2,
                2,
                1,
                2,
            ],
            [
                1,
                64,
                5,
                1,
                6,
                8,
                8,
                64,
                8,
                8,
                2,
                2,
                1,
                2,
            ],
            [
                2,
                96,
                5,
                1,
                6,
                8,
                8,
                96,
                8,
                8,
                2,
                2,
                1,
                2,
            ],
            [
                1,
                128,
                3,
                1,
                6,
                12,
                12,
                128,
                8,
                8,
                2,
                2,
                1,
                2,
            ],
            [1, 768, 3, 1, 6, 16, 16, 0, 0, 0, 2, 2, 1, 2],
        ],
    ],
    "M3": [
        12,
        [4, 3],
        1024,
        [1.0, 0.5],
        [0.0, 0.5],
        [1, 3, 8, 12],
        [16, 24, 80, 864],
        [
            # s,  c, ks, c1, c2, g1, g2, c3, g3, g4, y1, y2, y3, r
            [
                2,
                16,
                3,
                2,
                2,
                0,
                12,
                16,
                4,
                4,
                0,
                2,
                0,
                1,
            ],
            [
                2,
                24,
                3,
                2,
                2,
                0,
                16,
                24,
                4,
                4,
                0,
                2,
                0,
                1,
            ],
            [
                1,
                24,
                3,
                2,
                2,
                0,
                24,
                24,
                4,
                4,
                0,
                2,
                0,
                1,
            ],
            [
                2,
                32,
                5,
                1,
                6,
                6,
                6,
                32,
                4,
                4,
                0,
                2,
                0,
                1,
            ],
            [
                1,
                32,
                5,
                1,
                6,
                8,
                8,
                32,
                4,
                4,
                0,
                2,
                0,
                2,
            ],
            [
                1,
                64,
                5,
                1,
                6,
                8,
                8,
                48,
                8,
                8,
                0,
                2,
                0,
                2,
            ],
            [
                1,
                80,
                5,
                1,
                6,
                8,
                8,
                80,
                8,
                8,
                0,
                2,
                0,
                2,
            ],
            [
                1,
                80,
                5,
                1,
                6,
                10,
                10,
                80,
                8,
                8,
                0,
                2,
                0,
                2,
            ],
            [
                2,
                120,
                5,
                1,
                6,
                10,
                10,
                120,
                10,
                10,
                0,
                2,
                0,
                2,
            ],
            [
                1,
                120,
                5,
                1,
                6,
                12,
                12,
                120,
                10,
                10,
                0,
                2,
                0,
                2,
            ],
            [
                1,
                144,
                3,
                1,
                6,
                12,
                12,
                144,
                12,
                12,
                0,
                2,
                0,
                2,
            ],
            [1, 864, 3, 1, 6, 12, 12, 0, 0, 0, 0, 2, 0, 2],
        ],
    ],
}
