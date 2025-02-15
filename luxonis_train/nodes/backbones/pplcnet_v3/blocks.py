import torch
from rich import print
from torch import Tensor, nn

from luxonis_train.nodes.blocks import (
    GeneralReparametrizableBlock,
    SqueezeExciteBlock,
)
from luxonis_train.nodes.blocks.blocks import ConvModule


class AffineActivation(nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = nn.Hardswish()
        # WARN: Is this correct?
        self.affine = AffineBlock()

    def forward(self, x: Tensor) -> Tensor:
        return self.affine(self.activation(x))


class AffineBlock(nn.Module):
    def __init__(self, scale_value: float = 1.0, bias_value: float = 0.0):
        super().__init__()

        self.scale = nn.Parameter(torch.full((1,), scale_value))
        self.bias = nn.Parameter(torch.full((1,), bias_value))

    def forward(self, x: Tensor) -> Tensor:
        return self.scale * x + self.bias


class LearnableRepLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        num_branches: int = 1,
    ):
        super().__init__()
        self.groups = groups
        self.stride = stride
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_conv_branches = num_branches
        self.padding = (kernel_size - 1) // 2

        self.identity = (
            nn.BatchNorm2d(
                num_features=in_channels,
            )
            if out_channels == in_channels and stride == 1
            else None
        )

        print(
            "conv_1x1",
            {
                "in_channels": in_channels,
                "out_channels": out_channels,
                "kernel_size": 1,
                "padding": 0,
                "stride": self.stride,
                "groups": groups,
                "activation": False,
            },
        )
        self.conv_1x1 = (
            ConvModule(
                in_channels,
                out_channels,
                1,
                self.stride,
                groups=groups,
                activation=False,
            )
            if kernel_size > 1
            else None
        )

        print(
            "conv_kxk",
            {
                "in_channels": in_channels,
                "out_channels": out_channels,
                "kernel_size": self.kernel_size,
                "stride": self.stride,
                "padding": (kernel_size - 1) // 2,
                "groups": groups,
                "activation": False,
            },
        )
        self.conv_kxk = nn.ModuleList(
            [
                ConvModule(
                    in_channels,
                    out_channels,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=(kernel_size - 1) // 2,
                    groups=groups,
                    activation=False,
                )
                for _ in range(self.num_conv_branches)
            ]
        )

        self.lab = AffineBlock()
        self.act = AffineActivation()

    def forward(self, x: Tensor) -> Tensor:
        out = 0
        if self.identity is not None:
            out += self.identity(x)

        if self.conv_1x1 is not None:
            out += self.conv_1x1(x)

        for conv in self.conv_kxk:
            out += conv(x)

        out = self.lab(out)
        if self.stride != 2:
            out = self.act(out)
        return out


class LCNetV3Block(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        use_se: bool = False,
        num_branches: int = 4,
    ):
        # blocks: list[nn.Module] = [
        #     LearnableRepLayer(
        #         in_channels=in_channels,
        #         out_channels=in_channels,
        #         kernel_size=kernel_size,
        #         stride=stride,
        #         # padding=(kernel_size - 1) // 2,
        #         groups=in_channels,
        #         num_branches=num_branches,
        #     )
        # ]
        # if use_se:
        #     blocks.append(
        #         SqueezeExciteBlock(
        #             in_channels=in_channels,
        #             intermediate_channels=in_channels // 4,
        #             approx_sigmoid=True,
        #         )
        #     )
        #
        # blocks.append(
        #     LearnableRepLayer(
        #         in_channels=in_channels,
        #         out_channels=out_channels,
        #         kernel_size=1,
        #         stride=1,
        #         num_branches=num_branches,
        #     )
        # )
        blocks: list[nn.Module] = [
            GeneralReparametrizableBlock(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=(kernel_size - 1) // 2,
                groups=in_channels,
                num_branches=num_branches,
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
                num_branches=num_branches,
                refine_block=AffineBlock(),
                activation=AffineActivation(),
            )
        )
        super().__init__(*blocks)
