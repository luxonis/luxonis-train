from typing import Any

from torch import nn

from luxonis_train.nodes.blocks import ConvModule, SqueezeExciteBlock


class MobileNetV3ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        use_se: bool = True,
        activation: Any = None,
        dilation: int = 1,
    ):
        super().__init__()
        self.if_shortcut = stride == 1 and in_channels == out_channels
        self.if_se = use_se

        act_module = self._get_activation_module(activation)

        self.expand_conv = ConvModule(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            activation=act_module,
            norm_momentum=0.9,
        )
        self.bottleneck_conv = ConvModule(
            in_channels=mid_channels,
            out_channels=mid_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=int((kernel_size - 1) // 2) * dilation,
            groups=mid_channels,
            activation=act_module,
            dilation=dilation,
            norm_momentum=0.9,
        )
        if self.if_se:
            self.mid_se = SqueezeExciteBlock(
                in_channels=mid_channels,
                intermediate_channels=mid_channels // 4,
                approx_sigmoid=True,
            )
        self.linear_conv = ConvModule(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            activation=False,
            norm_momentum=0.9,
        )

    def _get_activation_module(self, activation):
        """Convert string or class to activation module instance."""
        if activation is None:
            return nn.ReLU()
        elif activation is False:
            return False
        elif isinstance(activation, str):
            if activation == "hardswish":
                return nn.Hardswish()
            elif activation == "relu":
                return nn.ReLU()
            else:
                return nn.ReLU()  # Default to ReLU for unknown strings
        elif isinstance(activation, type) and issubclass(
            activation, nn.Module
        ):
            return activation()
        elif isinstance(activation, nn.Module):
            return activation
        else:
            return nn.ReLU()  # Default fallback

    def forward(self, x):
        identity = x
        x = self.expand_conv(x)
        x = self.bottleneck_conv(x)
        if self.if_se:
            x = self.mid_se(x)
        x = self.linear_conv(x)
        if self.if_shortcut:
            x += identity
        return x
