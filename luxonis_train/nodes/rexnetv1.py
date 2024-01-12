"""Implementation of the ReXNetV1 backbone.

Source: U{https://github.com/clovaai/rexnet}
@license: U{MIT<https://github.com/clovaai/rexnet/blob/master/LICENSE>}
"""


import torch
from torch import Tensor, nn

from luxonis_train.nodes.blocks import (
    ConvModule,
)
from luxonis_train.utils.general import make_divisible

from .base_node import BaseNode


class ReXNetV1_lite(BaseNode[Tensor, list[Tensor]]):
    attach_index: int = -1

    def __init__(
        self,
        fix_head_stem: bool = False,
        divisible_value: int = 8,
        input_ch: int = 16,
        final_ch: int = 164,
        multiplier: float = 1.0,
        kernel_sizes: int | list[int] = 3,
        **kwargs,
    ):
        """ReXNetV1_lite backbone.

        @type fix_head_stem: bool
        @param fix_head_stem: Whether to multiply head stem. Defaults to False.
        @type divisible_value: int
        @param divisible_value: Divisor used. Defaults to 8.
        @type input_ch: int
        @param input_ch: Starting channel dimension. Defaults to 16.
        @type final_ch: int
        @param final_ch: Final channel dimension. Defaults to 164.
        @type multiplier: float
        @param multiplier: Channel dimension multiplier. Defaults to 1.0.
        @type kernel_sizes: int | list[int]
        @param kernel_sizes: Kernel size for each block. Defaults to 3.
        """
        super().__init__(**kwargs)

        self.out_indices = [1, 4, 10, 16]
        self.channels = [16, 48, 112, 184]
        layers = [1, 2, 2, 3, 3, 5]
        strides = [1, 2, 2, 2, 1, 2]

        kernel_sizes = (
            [kernel_sizes] * 6 if isinstance(kernel_sizes, int) else kernel_sizes
        )

        strides = sum(
            [
                [element] + [1] * (layers[idx] - 1)
                for idx, element in enumerate(strides)
            ],
            [],
        )
        ts = [1] * layers[0] + [6] * sum(layers[1:])
        kernel_sizes = sum(
            [[element] * layers[idx] for idx, element in enumerate(kernel_sizes)], []
        )
        self.num_convblocks = sum(layers[:])

        features: list[nn.Module] = []
        inplanes = input_ch / multiplier if multiplier < 1.0 else input_ch
        first_channel = 32 / multiplier if multiplier < 1.0 or fix_head_stem else 32
        first_channel = make_divisible(
            int(round(first_channel * multiplier)), divisible_value
        )

        in_channels_group = []
        channels_group = []

        features.append(
            ConvModule(
                3,
                first_channel,
                kernel_size=3,
                stride=2,
                padding=1,
                activation=nn.ReLU6(inplace=True),
            )
        )

        for i in range(self.num_convblocks):
            inplanes_divisible = make_divisible(
                int(round(inplanes * multiplier)), divisible_value
            )
            if i == 0:
                in_channels_group.append(first_channel)
                channels_group.append(inplanes_divisible)
            else:
                in_channels_group.append(inplanes_divisible)
                inplanes += final_ch / (self.num_convblocks - 1 * 1.0)
                inplanes_divisible = make_divisible(
                    int(round(inplanes * multiplier)), divisible_value
                )
                channels_group.append(inplanes_divisible)

        assert channels_group
        for in_c, c, t, k, s in zip(
            in_channels_group, channels_group, ts, kernel_sizes, strides, strict=True
        ):
            features.append(
                LinearBottleneck(
                    in_channels=in_c, channels=c, t=t, kernel_size=k, stride=s
                )
            )

        pen_channels = (
            int(1280 * multiplier) if multiplier > 1 and not fix_head_stem else 1280
        )
        features.append(
            ConvModule(
                in_channels=c,  # type: ignore
                out_channels=pen_channels,
                kernel_size=1,
                activation=nn.ReLU6(inplace=True),
            )
        )
        self.features = nn.Sequential(*features)

    def forward(self, x: Tensor) -> list[Tensor]:
        outs = []
        for i, m in enumerate(self.features):
            x = m(x)
            if i in self.out_indices:
                outs.append(x)
        return outs


class LinearBottleneck(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: int,
        t: int,
        kernel_size: int = 3,
        stride: int = 1,
        **kwargs,
    ):
        super(LinearBottleneck, self).__init__(**kwargs)
        self.conv_shortcut = None
        self.use_shortcut = stride == 1 and in_channels <= channels
        self.in_channels = in_channels
        self.out_channels = channels
        out = []
        if t != 1:
            dw_channels = in_channels * t
            out.append(
                ConvModule(
                    in_channels=in_channels,
                    out_channels=dw_channels,
                    kernel_size=1,
                    activation=nn.ReLU6(inplace=True),
                )
            )
        else:
            dw_channels = in_channels
        out.append(
            ConvModule(
                in_channels=dw_channels,
                out_channels=dw_channels * 1,
                kernel_size=kernel_size,
                stride=stride,
                padding=(kernel_size // 2),
                groups=dw_channels,
                activation=nn.ReLU6(inplace=True),
            )
        )
        out.append(
            ConvModule(
                in_channels=dw_channels,
                out_channels=channels,
                kernel_size=1,
                activation=nn.Identity(),
            )
        )

        self.out = nn.Sequential(*out)

    def forward(self, x):
        out = self.out(x)

        if self.use_shortcut:
            # this results in a ScatterND node which isn't supported yet in myriad
            # out[:, 0:self.in_channels] += x
            a = out[:, : self.in_channels]
            b = x
            a = a + b
            c = out[:, self.in_channels :]
            d = torch.concat([a, c], dim=1)
            return d

        return out
