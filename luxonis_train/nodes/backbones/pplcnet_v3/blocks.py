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
        # WARN: Is this correct?
        self.affine = AffineBlock()

    def forward(self, x: Tensor) -> Tensor:
        return self.affine(self.activation(x))


class AffineBlock(nn.Module):
    @typechecked
    def __init__(self, scale_value: float = 1.0, bias_value: float = 0.0):
        super().__init__()

        self.scale = nn.Parameter(torch.full((1,), scale_value))
        self.bias = nn.Parameter(torch.full((1,), bias_value))

    def forward(self, x: Tensor) -> Tensor:
        return self.scale * x + self.bias


class LCNetV3Block(nn.Sequential):
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
        blocks: list[nn.Module] = [
            GeneralReparametrizableBlock(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=(kernel_size - 1) // 2,
                groups=in_channels,
                n_branches=n_branches,
                refine_block=AffineBlock(),
                activation=AffineActivation() if stride != 2 else False,
            )
        ]
        if use_se:
            blocks.append(
                SqueezeExciteBlock(
                    in_channels=in_channels,
                    intermediate_channels=in_channels // 4,
                    approx_sigmoid=True,
                )
            )

        blocks.append(
            GeneralReparametrizableBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                padding=0,
                kernel_size=1,
                stride=1,
                n_branches=n_branches,
                refine_block=AffineBlock(),
                activation=AffineActivation(),
            )
        )
        super().__init__(*blocks)
