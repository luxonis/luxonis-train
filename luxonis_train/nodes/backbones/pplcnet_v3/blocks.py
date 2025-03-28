import torch
from torch import Tensor, nn
from torch.nn import (
    BatchNorm2d,
    Conv2d,
    ReLU,
)
from torch.nn import functional as F

from luxonis_train.nodes.blocks import ConvModule, SqueezeExciteBlock


def make_divisible(
    v: float, divisor: int = 16, min_value: int | None = None
) -> int:
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class Act(nn.Module):
    def __init__(self, act: str = "hswish"):
        super().__init__()
        if act == "hswish":
            self.act = lambda x: x * F.relu6(x + 3) * (1.0 / 6)
        else:
            assert act == "relu"
            self.act = ReLU()
        self.lab = LearnableAffineBlock()

    def forward(self, x: Tensor) -> Tensor:
        return self.lab(self.act(x))


class LearnableAffineBlock(nn.Module):
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
        num_conv_branches: int = 1,
    ):
        super().__init__()
        self.groups = groups
        self.stride = stride
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_conv_branches = num_conv_branches
        self.padding = (kernel_size - 1) // 2

        self.identity = (
            BatchNorm2d(
                num_features=in_channels,
            )
            if out_channels == in_channels and stride == 1
            else None
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
                    activation=nn.Identity(),
                )
                for _ in range(self.num_conv_branches)
            ]
        )

        self.conv_1x1 = (
            ConvModule(
                in_channels,
                out_channels,
                1,
                self.stride,
                groups=groups,
                activation=nn.Identity(),
            )
            if kernel_size > 1
            else None
        )

        self.lab = LearnableAffineBlock()
        self.act = Act()

    def forward(self, x: Tensor) -> Tensor:
        if hasattr(self, "reparam_conv"):
            out = self.lab(self.reparam_conv(x))
            if self.stride != 2:
                out = self.act(out)
            return out

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

    def reparametrize(self) -> None:
        if hasattr(self, "reparam_conv"):
            return
        kernel, bias = self._get_kernel_bias()
        self.reparam_conv = Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            groups=self.groups,
            bias=True,
        )
        self.reparam_conv.weight.data = kernel  # type: ignore
        self.reparam_conv.bias.data = bias  # type: ignore

        del self.conv_kxk
        del self.conv_1x1
        if hasattr(self, "identity"):
            del self.identity
        if hasattr(self, "id_tensor"):
            del self.id_tensor

    def _pad_kernel_1x1_to_kxk(
        self, kernel1x1: Tensor | None, pad: int
    ) -> Tensor | int:
        if kernel1x1 is None or torch.equal(
            kernel1x1, torch.tensor(0, device=kernel1x1.device)
        ):
            return torch.tensor(0)
        else:
            return nn.functional.pad(kernel1x1, [pad, pad, pad, pad])

    def _get_kernel_bias(self) -> tuple[Tensor, Tensor]:
        device = next(self.parameters()).device
        kernel_conv_1x1, bias_conv_1x1 = self._fuse_bn_tensor(
            self.conv_1x1, device
        )

        kernel_conv_1x1 = self._pad_kernel_1x1_to_kxk(
            kernel_conv_1x1,
            self.kernel_size // 2,  # type: ignore
        )

        kernel_identity, bias_identity = self._fuse_bn_tensor(
            self.identity, device
        )

        kernel_conv_kxk = 0
        bias_conv_kxk = 0
        for conv in self.conv_kxk:
            kernel, bias = self._fuse_bn_tensor(conv, device)
            kernel_conv_kxk += kernel
            bias_conv_kxk += bias

        kernel_reparam = kernel_conv_kxk + kernel_conv_1x1 + kernel_identity
        bias_reparam = bias_conv_kxk + bias_conv_1x1 + bias_identity
        return kernel_reparam, bias_reparam

    def _fuse_bn_tensor(
        self, branch: nn.Module | None, device: torch.device
    ) -> tuple[Tensor, Tensor]:
        if branch is None:
            return torch.tensor(0), torch.tensor(0)
        elif isinstance(branch, ConvModule):
            kernel = branch[0].weight
            running_mean = branch[1].running_mean
            running_var = branch[1].running_var
            gamma = branch[1].weight
            beta = branch[1].bias
            eps = branch[1].eps
        else:
            assert isinstance(branch, BatchNorm2d)
            if not hasattr(self, "id_tensor"):
                input_dim = self.in_channels // self.groups
                kernel_value = torch.zeros(
                    (
                        self.in_channels,
                        input_dim,
                        self.kernel_size,
                        self.kernel_size,
                    ),
                    dtype=branch.weight.dtype,
                    device=device,
                )
                for i in range(self.in_channels):
                    kernel_value[
                        i,
                        i % input_dim,
                        self.kernel_size // 2,
                        self.kernel_size // 2,
                    ] = 1
                self.id_tensor = kernel_value
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        assert running_var is not None
        std = (running_var + eps).sqrt()  # type: ignore
        t = (gamma / std).reshape((-1, 1, 1, 1)).to(kernel.device)  # type: ignore
        return kernel * t, beta - running_mean * gamma / std  # type: ignore


class LCNetV3Block(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        dw_size: int,
        use_se: bool = False,
        conv_kxk_num: int = 4,
    ):
        super().__init__()
        self.use_se = use_se
        self.dw_conv = LearnableRepLayer(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=dw_size,
            stride=stride,
            groups=in_channels,
            num_conv_branches=conv_kxk_num,
        )
        if use_se:
            self.se = SqueezeExciteBlock(
                in_channels=in_channels,
                intermediate_channels=in_channels // 4,
                approx_sigmoid=True,
            )

        self.pw_conv = LearnableRepLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            num_conv_branches=conv_kxk_num,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.dw_conv(x)
        if self.use_se:
            x = self.se(x)
        x = self.pw_conv(x)
        return x
