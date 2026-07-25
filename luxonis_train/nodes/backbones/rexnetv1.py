import torch
from torch import Tensor, nn

from luxonis_train.nodes.base_node import BaseNode
from luxonis_train.nodes.blocks import ConvBlock
from luxonis_train.utils import make_divisible


class ReXNetV1_lite(BaseNode):
    def __init__(
        self,
        fix_head_stem: bool = False,
        divisible_value: int = 8,
        input_ch: int = 16,
        final_ch: int = 164,
        multiplier: float = 1.0,
        kernel_sizes: int | list[int] = 3,
        out_indices: list[int] | None = None,
        **kwargs,
    ):
        """ReXNetV1 (Rank Expansion Networks) backbone, lite version.

        ReXNet proposes a new approach to designing lightweight CNN architectures by:

            - Studying proper channel dimension expansion at the layer level using rank analysis
            - Searching for effective channel configurations across the entire network
            - Parameterizing channel dimensions as a linear function of network depth

        Key aspects:

            - Uses inverted bottleneck blocks similar to MobileNetV2
            - Employs a linear parameterization of channel dimensions across blocks
            - Replaces ReLU6 with SiLU (Swish-1) activation in certain layers
            - Incorporates Squeeze-and-Excitation modules

        ReXNet achieves state-of-the-art performance among lightweight models on ImageNet
        classification and transfers well to tasks like object detection and fine-grained classification.

        Source: U{https://github.com/clovaai/rexnet}

        @license: U{MIT
            <https://github.com/clovaai/rexnet/blob/master/LICENSE>}
        @copyright: 2021-present NAVER Corp.
        @see U{Rethinking Channel Dimensions for Efficient Model Design <https://arxiv.org/abs/2007.00992>}
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
        @param out_indices: list[int] | None
        @param out_indices: Indices of the output layers. Defaults to [1, 4, 10, 17].
        """
        super().__init__(**kwargs)

        layers = [1, 2, 2, 3, 3, 5]
        strides = [1, 2, 2, 2, 1, 2]

        self.n_convblocks = sum(layers)
        self.out_indices = out_indices or [1, 4, 10, 17]

        strides, ts, kernel_sizes = self._expand_layer_configs(
            layers, strides, kernel_sizes
        )

        inplanes, first_channel = self._compute_stem_channels(
            input_ch, multiplier, fix_head_stem, divisible_value
        )
        in_channels_group, channels_group = self._compute_channel_groups(
            self.n_convblocks,
            first_channel,
            inplanes,
            multiplier,
            final_ch,
            divisible_value,
        )
        assert channels_group

        pen_channels = (
            int(1280 * multiplier)
            if multiplier > 1 and not fix_head_stem
            else 1280
        )

        features: list[nn.Module] = [
            ConvBlock(
                3,
                first_channel,
                kernel_size=3,
                stride=2,
                padding=1,
                activation=nn.ReLU6(inplace=True),
            )
        ]
        for in_c, c, t, k, s in zip(
            in_channels_group,
            channels_group,
            ts,
            kernel_sizes,
            strides,
            strict=True,
        ):
            features.append(
                LinearBottleneck(
                    in_channels=in_c, channels=c, t=t, kernel_size=k, stride=s
                )
            )
        features.append(
            ConvBlock(
                in_channels=channels_group[-1],
                out_channels=pen_channels,
                kernel_size=1,
                activation=nn.ReLU6(inplace=True),
            )
        )
        self.features = nn.Sequential(*features)

    @staticmethod
    def _expand_layer_configs(
        layers: list[int],
        strides: list[int],
        kernel_sizes: int | list[int],
    ) -> tuple[list[int], list[int], list[int]]:
        """Expand the per-stage configs into per-block lists."""
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * 6

        expanded_strides = [
            s if i == 0 else 1
            for layer, s in zip(layers, strides, strict=True)
            for i in range(layer)
        ]
        ts = [1] * layers[0] + [6] * sum(layers[1:])
        expanded_kernel_sizes = [
            ks
            for ks, layer in zip(kernel_sizes, layers, strict=True)
            for _ in range(layer)
        ]
        return expanded_strides, ts, expanded_kernel_sizes

    @staticmethod
    def _compute_stem_channels(
        input_ch: int,
        multiplier: float,
        fix_head_stem: bool,
        divisible_value: int,
    ) -> tuple[float, int]:
        inplanes = input_ch / multiplier if multiplier < 1.0 else input_ch
        first_channel = (
            32 / multiplier if multiplier < 1.0 or fix_head_stem else 32
        )
        first_channel = make_divisible(
            round(first_channel * multiplier), divisible_value
        )
        return inplanes, first_channel

    @staticmethod
    def _compute_channel_groups(
        n_convblocks: int,
        first_channel: int,
        inplanes: float,
        multiplier: float,
        final_ch: int,
        divisible_value: int,
    ) -> tuple[list[int], list[int]]:
        in_channels_group: list[int] = []
        channels_group: list[int] = []
        for i in range(n_convblocks):
            inplanes_divisible = make_divisible(
                round(inplanes * multiplier), divisible_value
            )
            if i == 0:
                in_channels_group.append(first_channel)
                channels_group.append(inplanes_divisible)
            else:
                in_channels_group.append(inplanes_divisible)
                inplanes += final_ch / (n_convblocks - 1 * 1.0)
                inplanes_divisible = make_divisible(
                    round(inplanes * multiplier), divisible_value
                )
                channels_group.append(inplanes_divisible)
        return in_channels_group, channels_group

    def forward(self, inputs: Tensor) -> list[Tensor]:
        outs: list[Tensor] = []
        for i, module in enumerate(self.features):
            inputs = module(inputs)
            if i in self.out_indices:
                outs.append(inputs)
        return outs


class LinearBottleneck(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: int,
        t: int,
        kernel_size: int = 3,
        stride: int = 1,
    ):
        super().__init__()
        self.use_shortcut = stride == 1 and in_channels <= channels
        self.in_channels = in_channels
        self.out_channels = channels
        out: list[nn.Module] = []
        if t != 1:
            dw_channels = in_channels * t
            out.append(
                ConvBlock(
                    in_channels=in_channels,
                    out_channels=dw_channels,
                    kernel_size=1,
                    activation=nn.ReLU6(inplace=True),
                )
            )
        else:
            dw_channels = in_channels
        out.append(
            ConvBlock(
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
            ConvBlock(
                in_channels=dw_channels,
                out_channels=channels,
                kernel_size=1,
                activation=None,
            )
        )

        self.out = nn.Sequential(*out)

    def forward(self, x: Tensor) -> Tensor:
        out = self.out(x)

        if self.use_shortcut:
            # NOTE: this results in a ScatterND node which isn't supported yet in myriad
            a = out[:, : self.in_channels]
            b = x
            a = a + b
            c = out[:, self.in_channels :]
            return torch.concat([a, c], dim=1)

        return out
