from typing import Literal

import torch
from torch import Tensor, nn

from luxonis_train.nodes.blocks import ConvBlock


class MicroBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        expand_ratio: tuple[int, int] = (2, 2),
        groups_1: tuple[int, int] = (0, 6),
        groups_2: tuple[int, int] = (1, 1),
        dy_shift: tuple[int, int, int] = (2, 0, 1),
        reduction_factor: int = 1,
        init_a: tuple[float, float] = (1.0, 1.0),
        init_b: tuple[float, float] = (0.0, 0.0),
    ):
        """MicroBlock: The basic building block of MicroNet.

        This block implements the Micro-Factorized Convolution and
        Dynamic Shift-Max activation. It can be configured to use
        different combinations of these components based on the network
        design.

        @type in_channels: int
        @param in_channels: Number of input channels.
        @type out_channels: int
        @param out_channels: Number of output channels.
        @type kernel_size: int
        @param kernel_size: Size of the convolution kernel. Defaults to
            3.
        @type stride: int
        @param stride: Stride of the convolution. Defaults to 1.
        @type expansion_ratios: tuple[int, int]
        @param expansion_ratios: Expansion ratios for the intermediate
            channels. Defaults to (2, 2).
        @type groups_1: tuple[int, int]
        @param groups_1: Groups for the first set of convolutions.
            Defaults to (0, 6).
        @type groups_2: tuple[int, int]
        @param groups_2: Groups for the second set of convolutions.
            Defaults to (1, 1).
        @type use_dynamic_shift: tuple[int, int, int]
        @param use_dynamic_shift: Flags to use Dynamic Shift-Max in
            different positions. Defaults to (2, 0, 1).
        @type reduction_factor: int
        @param reduction_factor: Reduction factor for the squeeze-and-
            excitation-like operation. Defaults to 1.
        @type init_a: tuple[float, float]
        @param init_a: Initialization parameters for Dynamic Shift-Max.
            Defaults to (1.0, 1.0).
        @type init_b: tuple[float, float]
        @param init_b: Initialization parameters for Dynamic Shift-Max.
            Defaults to (0.0, 0.0).
        """
        super().__init__()

        self.use_residual = stride == 1 and in_channels == out_channels
        self.expand_ratio = expand_ratio
        use_dy1, use_dy2, use_dy3 = dy_shift
        group1, group2 = groups_2
        reduction = 8 * reduction_factor
        intermediate_channels = in_channels * expand_ratio[0] * expand_ratio[1]

        if groups_1[0] == 0:
            self.layers = self._create_lite_block(
                in_channels,
                out_channels,
                intermediate_channels,
                kernel_size,
                stride,
                groups_1[1],
                group1,
                group2,
                use_dy2,
                use_dy3,
                reduction,
                init_a,
                init_b,
            )
        elif group2 == 0:
            self.layers = self._create_transition_block(
                in_channels,
                intermediate_channels,
                groups_1[0],
                groups_1[1],
                use_dy3,
                reduction,
            )
        else:
            self.layers = self._create_full_block(
                in_channels,
                out_channels,
                intermediate_channels,
                kernel_size,
                stride,
                groups_1,
                group1,
                group2,
                use_dy1,
                use_dy2,
                use_dy3,
                reduction,
                init_a,
                init_b,
            )

    def forward(self, inputs: Tensor) -> Tensor:
        out = self.layers(inputs)
        if self.use_residual:
            out += inputs
        return out

    def _create_lite_block(
        self,
        in_channels: int,
        out_channels: int,
        intermediate_channels: int,
        kernel_size: int,
        stride: int,
        group1: int,
        group2: int,
        group3: int,
        use_dy2: int,
        use_dy3: int,
        reduction: int,
        init_a: tuple[float, float],
        init_b: tuple[float, float],
    ) -> nn.Sequential:
        return nn.Sequential(
            DepthSpatialSepConv(
                in_channels, self.expand_ratio, kernel_size, stride
            ),
            DYShiftMax(
                intermediate_channels,
                intermediate_channels,
                init_a,
                init_b,
                use_dy2 == 2,
                group1,
                reduction,
            )
            if use_dy2 > 0
            else nn.ReLU6(True),
            ChannelShuffle(group1),
            ChannelShuffle(intermediate_channels // 2)
            if use_dy2 != 0
            else nn.Sequential(),
            ConvBlock(
                in_channels=intermediate_channels,
                out_channels=out_channels,
                kernel_size=1,
                groups=group2,
                activation=None,
            ),
            DYShiftMax(
                out_channels,
                out_channels,
                (1.0, 0.0),
                (0.0, 0.0),
                False,
                group3,
                reduction // 2,
            )
            if use_dy3 > 0
            else nn.Sequential(),
            ChannelShuffle(group3),
            ChannelShuffle(out_channels // 2)
            if out_channels % 2 == 0 and use_dy3 != 0
            else nn.Sequential(),
        )

    def _create_transition_block(
        self,
        in_channels: int,
        intermediate_channels: int,
        group1: int,
        group2: int,
        use_dy3: int,
        reduction: int,
    ) -> nn.Sequential:
        return nn.Sequential(
            ConvBlock(
                in_channels=in_channels,
                out_channels=intermediate_channels,
                kernel_size=1,
                groups=group1,
                activation=None,
            ),
            DYShiftMax(
                intermediate_channels,
                intermediate_channels,
                (1.0, 0.0),
                (0.0, 0.0),
                False,
                group2,
                reduction,
            )
            if use_dy3 > 0
            else nn.Sequential(),
        )

    def _create_full_block(
        self,
        in_channels: int,
        out_channels: int,
        intermediate_channels: int,
        kernel_size: int,
        stride: int,
        groups_1: tuple[int, int],
        group1: int,
        group2: int,
        use_dy1: int,
        use_dy2: int,
        use_dy3: int,
        reduction: int,
        init_a: tuple[float, float],
        init_b: tuple[float, float],
    ) -> nn.Sequential:
        return nn.Sequential(
            ConvBlock(
                in_channels=in_channels,
                out_channels=intermediate_channels,
                kernel_size=1,
                groups=groups_1[0],
                activation=None,
            ),
            DYShiftMax(
                intermediate_channels,
                intermediate_channels,
                init_a,
                init_b,
                use_dy1 == 2,
                groups_1[1],
                reduction,
            )
            if use_dy1 > 0
            else nn.ReLU6(True),
            ChannelShuffle(groups_1[1]),
            DepthSpatialSepConv(
                intermediate_channels, (1, 1), kernel_size, stride
            ),
            DYShiftMax(
                intermediate_channels,
                intermediate_channels,
                init_a,
                init_b,
                use_dy2 == 2,
                groups_1[1],
                reduction,
                True,
            )
            if use_dy2 > 0
            else nn.ReLU6(True),
            ChannelShuffle(intermediate_channels // 4)
            if use_dy1 != 0 and use_dy2 != 0
            else nn.Sequential()
            if use_dy1 == 0 and use_dy2 == 0
            else ChannelShuffle(intermediate_channels // 2),
            ConvBlock(
                in_channels=intermediate_channels,
                out_channels=out_channels,
                kernel_size=1,
                groups=group1,
                activation=None,
            ),
            DYShiftMax(
                out_channels,
                out_channels,
                (1.0, 0.0),
                (0.0, 0.0),
                False,
                group2,
                reduction=reduction // 2
                if out_channels < intermediate_channels
                else reduction,
            )
            if use_dy3 > 0
            else nn.Sequential(),
            ChannelShuffle(group2),
            ChannelShuffle(out_channels // 2)
            if use_dy3 != 0
            else nn.Sequential(),
        )


class ChannelShuffle(nn.Module):
    def __init__(self, groups: int):
        """Shuffle the channels of the input tensor.

        This operation is used to mix information between groups after
        grouped convolutions.

        @type groups: int
        @param groups: Number of groups to divide the channels into
            before shuffling.
        """
        super().__init__()
        self.groups = groups

    def forward(self, x: Tensor) -> Tensor:
        batch_size, channels, height, width = x.size()
        channels_per_group = channels // self.groups
        x = x.view(batch_size, self.groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        return x.view(batch_size, -1, height, width)


class DYShiftMax(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        init_a: tuple[float, float] = (0.0, 0.0),
        init_b: tuple[float, float] = (0.0, 0.0),
        use_relu: bool = True,
        groups: int = 6,
        reduction: int = 4,
        expansion: bool = False,
    ):
        """Dynamic Shift-Max activation function.

        This module implements the Dynamic Shift-Max operation, which
        adaptively fuses and selects channel information based on the
        input.

        @type in_channels: int
        @param in_channels: Number of input channels.
        @type out_channels: int
        @param out_channels: Number of output channels.
        @type init_a: tuple[float, float]
        @param init_a: Initial values for the 'a' parameters. Defaults
            to (0.0, 0.0).
        @type init_b: tuple[float, float]
        @param init_b: Initial values for the 'b' parameters. Defaults
            to (0.0, 0.0).
        @type use_relu: bool
        @param use_relu: Whether to use ReLU activation. Defaults to
            True.
        @type groups: int
        @param groups: Number of groups for channel shuffling. Defaults
            to 6.
        @type reduction: int
        @param reduction: Reduction factor for the squeeze operation.
            Defaults to 4.
        @type expansion: bool
        @param expansion: Whether to use expansion in grouping. Defaults
            to False.
        """
        super().__init__()
        self.exp: Literal[2, 4] = 4 if use_relu else 2
        self.init_a = init_a
        self.init_b = init_b
        self.out_channels = out_channels

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        squeeze_channels = _make_divisible(in_channels // reduction, 4)

        self.fc = nn.Sequential(
            nn.Linear(in_channels, squeeze_channels),
            nn.ReLU(),
            nn.Linear(squeeze_channels, out_channels * self.exp),
            nn.Hardsigmoid(),
        )

        if groups != 1 and expansion:
            groups = in_channels // groups

        channels_per_group = in_channels // groups
        index = torch.arange(in_channels).view(1, in_channels, 1, 1)
        index = index.view(1, groups, channels_per_group, 1, 1)
        index_groups = torch.split(index, [1, groups - 1], dim=1)
        index_groups = torch.cat([index_groups[1], index_groups[0]], dim=1)
        index_splits = torch.split(
            index_groups, [1, channels_per_group - 1], dim=2
        )
        index_splits = torch.cat([index_splits[1], index_splits[0]], dim=2)
        self.index = index_splits.view(in_channels).long()

    def forward(self, x: Tensor) -> Tensor:
        batch_size, channels, _, _ = x.shape
        x_out = x

        y = self.avg_pool(x).view(batch_size, channels)
        y: Tensor = self.fc(y).view(batch_size, -1, 1, 1)
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
            a1, b1 = y.split(self.out_channels, dim=1)
            a1 = a1 + self.init_a[0]
            b1 = b1 + self.init_b[0]
            out = x_out * a1 + x2 * b1
        else:
            raise RuntimeError("Expansion should be 2 or 4.")

        return out


def _make_divisible(value: int, divisor: int) -> int:
    min_value = divisor
    new_v = max(min_value, int(value + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * value:
        new_v += divisor
    return new_v


class SpatialSepConvSF(nn.Module):
    def __init__(
        self,
        in_channels: int,
        outs: tuple[int, int],
        kernel_size: int,
        stride: int,
    ):
        super().__init__()
        out_channels1, out_channels2 = outs
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels1,
                kernel_size=(kernel_size, 1),
                stride=(stride, 1),
                padding=(kernel_size // 2, 0),
                bias=False,
            ),
            nn.BatchNorm2d(out_channels1),
            nn.Conv2d(
                out_channels1,
                out_channels1 * out_channels2,
                kernel_size=(1, kernel_size),
                stride=(1, stride),
                padding=(0, kernel_size // 2),
                groups=out_channels1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels1 * out_channels2),
            ChannelShuffle(out_channels1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class Stem(nn.Module):
    def __init__(
        self, in_channels: int, stride: int, outs: tuple[int, int] = (4, 4)
    ):
        super().__init__()
        self.stem = nn.Sequential(
            SpatialSepConvSF(in_channels, outs, 3, stride), nn.ReLU6(True)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.stem(x)


class DepthSpatialSepConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        expand: tuple[int, int],
        kernel_size: int,
        stride: int,
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
                padding=(kernel_size // 2, 0),
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(intermediate_channels),
            nn.Conv2d(
                intermediate_channels,
                out_channels,
                (1, kernel_size),
                (1, stride),
                padding=(0, kernel_size // 2),
                groups=intermediate_channels,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)
