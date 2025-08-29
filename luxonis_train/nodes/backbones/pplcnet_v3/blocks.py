import torch
from torch import Tensor, nn
from typeguard import typechecked

from luxonis_train.nodes.blocks import (
    GeneralReparametrizableBlock,
    SqueezeExciteBlock,
)


class AffineActivation(nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = nn.Hardswish()
        self.affine = AffineBlock()

    def forward(self, x: Tensor) -> Tensor:
        # WARN: Is the order correct (activation -> affine)?
        return self.affine(self.activation(x))


class AffineBlock(nn.Module):
    @typechecked
    def __init__(self, scale_value: float = 1.0, bias_value: float = 0.0):
        super().__init__()

        self.scale = nn.Parameter(torch.full((1,), scale_value))
        self.bias = nn.Parameter(torch.full((1,), bias_value))

    def forward(self, x: Tensor) -> Tensor:
        return self.scale * x + self.bias


class LCNetV3Block(nn.Module):
    @typechecked
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        use_se: bool = False,
        n_branches: int = 4,
    ):
        super().__init__()
        self.dw_conv = GeneralReparametrizableBlock(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            groups=in_channels,
            n_branches=n_branches,
            refine_block=AffineBlock(),
            activation=AffineActivation(),
        )
        if use_se:
            self.se = SqueezeExciteBlock(
                in_channels=in_channels,
                intermediate_channels=in_channels // 4,
                hard_sigmoid=True,
            )
        else:
            self.se = nn.Identity()

        self.pw_conv = GeneralReparametrizableBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            padding=0,
            kernel_size=1,
            stride=1,
            n_branches=n_branches,
            refine_block=AffineBlock(),
            activation=AffineActivation(),
            use_scale_layer=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.pw_conv(self.se(self.dw_conv(x)))


class LCNetV3Layer(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: list[int],
        kernel_sizes: list[int],
        strides: list[int],
        use_se: list[bool],
        n_branches: int = 4,
        scale: float = 1.0,
    ):
        self.in_channels = in_channels
        self.out_channels = scale_up(out_channels[-1], scale)
        layer = []
        for out_channel, kernel_size, stride, se in zip(
            out_channels,
            kernel_sizes,
            strides,
            use_se,
            strict=True,
        ):
            out_channel = scale_up(out_channel, scale)
            layer.append(
                LCNetV3Block(
                    in_channels=in_channels,
                    out_channels=out_channel,
                    kernel_size=kernel_size,
                    stride=stride,
                    use_se=se,
                    n_branches=n_branches,
                )
            )
            in_channels = out_channel
        super().__init__(*layer)


def scale_up(
    v: float, scale: float, divisor: int = 16, min_value: int | None = None
) -> int:
    v = v * scale
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v
