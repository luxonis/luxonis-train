import math
from typing import Literal, TypeVar

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from luxonis_train.nodes.activations import HSigmoid


class EfficientDecoupledBlock(nn.Module):
    def __init__(self, n_classes: int, in_channels: int):
        """Efficient Decoupled block used for class and regression
        predictions.

        @type n_classes: int
        @param n_classes: Number of classes.
        @type in_channels: int
        @param in_channels: Number of input channels.
        """
        super().__init__()

        self.decoder = ConvModule(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            activation=nn.SiLU(),
        )

        self.class_branch = nn.Sequential(
            ConvModule(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                activation=nn.SiLU(),
            ),
            nn.Conv2d(
                in_channels=in_channels, out_channels=n_classes, kernel_size=1
            ),
        )
        self.regression_branch = nn.Sequential(
            ConvModule(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                activation=nn.SiLU(),
            ),
            nn.Conv2d(in_channels=in_channels, out_channels=4, kernel_size=1),
        )

        prior_prob = 1e-2
        self._initialize_weights_and_biases(prior_prob)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        out_feature = self.decoder(x)

        out_cls = self.class_branch(out_feature)
        out_reg = self.regression_branch(out_feature)

        return out_feature, out_cls, out_reg

    def _initialize_weights_and_biases(self, prior_prob: float) -> None:
        data = [
            (self.class_branch[-1], -math.log((1 - prior_prob) / prior_prob)),
            (self.regression_branch[-1], 1.0),
        ]
        for module, fill_value in data:
            assert module.bias is not None
            b = module.bias.view(-1)
            b.data.fill_(fill_value)
            module.bias = nn.Parameter(b.view(-1), requires_grad=True)

            w = module.weight
            w.data.fill_(0.0)
            module.weight = nn.Parameter(w, requires_grad=True)


class SegProto(nn.Module):
    def __init__(
        self, in_channels: int, mid_channels: int = 256, out_channels: int = 32
    ):
        """Initializes the segmentation prototype generator.

        @type in_channels: int
        @param in_channels: Number of input channels.
        @type mid_channels: int
        @param mid_channels: Number of intermediate channels. Defaults
            to 256.
        @type out_channels: int
        @param out_channels: Number of output channels. Defaults to 32.
        """
        super().__init__()
        self.conv1 = ConvModule(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            activation=nn.SiLU(),
        )
        self.upsample = nn.ConvTranspose2d(
            in_channels=mid_channels,
            out_channels=mid_channels,
            kernel_size=2,
            stride=2,
            bias=True,
        )
        self.conv2 = ConvModule(
            in_channels=mid_channels,
            out_channels=mid_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            activation=nn.SiLU(),
        )
        self.conv3 = ConvModule(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            activation=nn.SiLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Defines the forward pass of the segmentation prototype
        generator.

        @type x: Tensor
        @param x: Input tensor.
        @rtype: Tensor
        @return: Processed tensor.
        """
        return self.conv3(self.conv2(self.upsample(self.conv1(x))))


class DFL(nn.Module):
    def __init__(self, reg_max: int = 16):
        """The DFL (Distribution Focal Loss) module processes input
        tensors by applying softmax over a specified dimension and
        projecting the resulting tensor to produce output logits.

        @type reg_max: int
        @param reg_max: Maximum number of regression outputs. Defaults
            to 16.
        """
        super().__init__()
        self.proj_conv = nn.Conv2d(reg_max, 1, kernel_size=1, bias=False)
        self.proj_conv.weight.data.copy_(
            torch.arange(reg_max, dtype=torch.float32).view(1, reg_max, 1, 1)
        )
        self.proj_conv.requires_grad_(False)

    def forward(self, x: Tensor) -> Tensor:
        bs, _, h, w = x.size()
        x = F.softmax(x.view(bs, 4, -1, h * w).permute(0, 2, 1, 3), dim=1)
        return self.proj_conv(x)[:, 0].view(bs, 4, h, w)


class ConvModule(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] | str = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = False,
        activation: nn.Module | None | Literal[False] = None,
        use_norm: bool = True,
        norm_momentum: float = 0.1,
    ):
        """Conv2d + Optional BN + Activation.

        @type in_channels: int
        @param in_channels: Number of input channels.
        @type out_channels: int
        @param out_channels: Number of output channels.
        @type kernel_size: int
        @param kernel_size: Kernel size.
        @type stride: int
        @param stride: Stride. Defaults to 1.
        @type padding: int | str
        @param padding: Padding. Defaults to 0.
        @type dilation: int
        @param dilation: Dilation. Defaults to 1.
        @type groups: int
        @param groups: Groups. Defaults to 1.
        @type bias: bool
        @param bias: Whether to use bias. Defaults to False.
        @type activation: L{nn.Module} | None | Literal[False]
        @param activation: Activation function. If None then nn.ReLU. If
            False then no activation. Defaults to None.
        @type use_norm: bool
        @param use_norm: Whether to use normalization. Defaults to True.
        """
        blocks: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
                bias,
            ),
        ]

        if use_norm:
            blocks.append(nn.BatchNorm2d(out_channels, momentum=norm_momentum))

        if activation is not False:
            blocks.append(activation or nn.ReLU())

        super().__init__(*blocks)


class DWConvModule(ConvModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = False,
        activation: nn.Module | None = None,
    ):
        """Depth-wise Conv2d + BN + Activation.

        @type in_channels: int
        @param in_channels: Number of input channels.
        @type out_channels: int
        @param out_channels: Number of output channels.
        @type kernel_size: int
        @param kernel_size: Kernel size.
        @type stride: int
        @param stride: Stride. Defaults to 1.
        @type padding: int
        @param padding: Padding. Defaults to 0.
        @type dilation: int
        @param dilation: Dilation. Defaults to 1.
        @type bias: bool
        @param bias: Whether to use bias. Defaults to False.
        @type activation: L{nn.Module} | None
        @param activation: Activation function. If None then nn.Relu.
        """
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=math.gcd(in_channels, out_channels),
            bias=bias,
            activation=activation,
        )


class UpBlock(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 2,
        stride: int = 2,
        upsample_mode: Literal["upsample", "conv_transpose"] = "upsample",
        inter_mode: str = "bilinear",
        align_corners: bool = False,
    ):
        """Upsampling with ConvTranspose2D or Upsample (based on the
        mode).

        @type in_channels: int
        @param in_channels: Number of input channels.
        @type out_channels: int
        @param out_channels: Number of output channels.
        @type kernel_size: int
        @param kernel_size: Kernel size. Defaults to C{2}.
        @type stride: int
        @param stride: Stride. Defaults to C{2}.
        @type upsample_mode: Literal["upsample", "conv_transpose"]
        @param upsample_mode: Upsampling method, either 'conv_transpose'
            (for ConvTranspose2D) or 'upsample' (for nn.Upsample).
        @type inter_mode: str
        @param inter_mode: Interpolation mode used for nn.Upsample
            (e.g., 'bilinear', 'nearest').
        @type align_corners: bool
        @param align_corners: Align corners option for upsampling
            methods that support it. Defaults to False.
        """
        layers = []

        if upsample_mode == "conv_transpose":
            layers.append(
                nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                )
            )
        elif upsample_mode == "upsample":
            layers.append(
                nn.Upsample(
                    scale_factor=stride,
                    mode=inter_mode,
                    align_corners=align_corners,
                )
            )
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))
        else:
            raise ValueError(
                "Unsupported upsample mode. Choose either 'conv_transpose' or 'upsample'."
            )

        layers.append(
            ConvModule(out_channels, out_channels, kernel_size=3, padding=1)
        )

        super().__init__(*layers)


class SqueezeExciteBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        intermediate_channels: int,
        approx_sigmoid: bool = False,
        activation: nn.Module | None = None,
    ):
        """Squeeze and Excite block,
        Adapted from U{Squeeze-and-Excitation Networks<https://arxiv.org/pdf/1709.01507.pdf>}.
        Code adapted from U{https://github.com/apple/ml-mobileone/blob/main/mobileone.py}.

        @type in_channels: int
        @param in_channels: Number of input channels.
        @type intermediate_channels: int
        @param intermediate_channels: Number of intermediate channels.
        @type approx_sigmoid: bool
        @param approx_sigmoid: Whether to use approximated sigmoid function. Defaults to False.
        @type activation: L{nn.Module} | None
        @param activation: Activation function. Defaults to L{nn.ReLU}.
        """
        super().__init__()

        activation = activation or nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv_down = nn.Conv2d(
            in_channels=in_channels,
            out_channels=intermediate_channels,
            kernel_size=1,
            bias=True,
        )
        self.activation = activation
        self.conv_up = nn.Conv2d(
            in_channels=intermediate_channels,
            out_channels=in_channels,
            kernel_size=1,
            bias=True,
        )
        self.sigmoid = HSigmoid() if approx_sigmoid else nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        weights = self.pool(x)
        weights = self.conv_down(weights)
        weights = self.activation(weights)
        weights = self.conv_up(weights)
        weights = self.sigmoid(weights)
        x = x * weights
        return x


class RepVGGBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        groups: int = 1,
        use_se: bool = False,
    ):
        """RepVGGBlock is a basic rep-style block, including training and deploy status
        This code is based on U{https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py}.


        @type in_channels: int
        @param in_channels: Number of input channels.
        @type out_channels: int
        @param out_channels: Number of output channels.
        @type kernel_size: int
        @param kernel_size: Kernel size. Defaults to C{3}.
        @type stride: int
        @param stride: Stride. Defaults to C{1}.
        @type padding: int
        @param padding: Padding. Defaults to C{1}.
        @type dilation: int
        @param dilation: Dilation. Defaults to C{1}.
        @type groups: int
        @param groups: Groups. Defaults to C{1}.
        @type padding_mode: str
        @param padding_mode: Padding mode. Defaults to C{"zeros"}.
        @type deploy: bool
        @param deploy: Whether to use deploy mode. Defaults to C{False}.
        @type use_se: bool
        @param use_se: Whether to use SqueezeExciteBlock. Defaults to C{False}.
        """
        super().__init__()

        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2

        self.nonlinearity = nn.ReLU()

        if use_se:
            # NOTE: that RepVGG-D2se uses SE before nonlinearity.
            # But RepVGGplus models uses SqueezeExciteBlock after nonlinearity.
            self.se = SqueezeExciteBlock(
                out_channels, intermediate_channels=int(out_channels // 16)
            )
        else:
            self.se = nn.Identity()

        self.rbr_identity = (
            nn.BatchNorm2d(num_features=in_channels)
            if out_channels == in_channels and stride == 1
            else None
        )
        self.rbr_dense = ConvModule(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            activation=False,
        )
        self.rbr_1x1 = ConvModule(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,
            padding=padding_11,
            groups=groups,
            activation=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        if hasattr(self, "rbr_reparam"):
            return self.nonlinearity(self.se(self.rbr_reparam(x)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(x)

        return self.nonlinearity(
            self.se(self.rbr_dense(x) + self.rbr_1x1(x) + id_out)
        )

    def reparametrize(self) -> None:
        if hasattr(self, "rbr_reparam"):
            return

        kernel, bias = self._get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(
            in_channels=self.rbr_dense[0].in_channels,
            out_channels=self.rbr_dense[0].out_channels,
            kernel_size=self.rbr_dense[0].kernel_size,
            stride=self.rbr_dense[0].stride,
            padding=self.rbr_dense[0].padding,
            dilation=self.rbr_dense[0].dilation,
            groups=self.rbr_dense[0].groups,
            bias=True,
        )
        self.rbr_reparam.weight.data = kernel  # type: ignore
        self.rbr_reparam.bias.data = bias  # type: ignore
        del self.rbr_dense
        del self.rbr_1x1
        if hasattr(self, "rbr_identity"):
            del self.rbr_identity
        if hasattr(self, "id_tensor"):
            del self.id_tensor

    def _get_equivalent_kernel_bias(self) -> tuple[Tensor, Tensor]:
        """Derives the equivalent kernel and bias in a DIFFERENTIABLE
        way."""
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return (
            kernel3x3
            + self._pad_1x1_to_3x3_tensor(kernel1x1)
            + kernelid.to(kernel3x3.device),
            bias3x3 + bias1x1 + biasid.to(bias3x3.device),
        )

    def _pad_1x1_to_3x3_tensor(self, kernel1x1: Tensor | None) -> Tensor:
        if kernel1x1 is None:
            return torch.tensor(0)
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(
        self, branch: nn.Module | None
    ) -> tuple[Tensor, Tensor]:
        if branch is None:
            return torch.tensor(0), torch.tensor(0)
        if isinstance(branch, nn.Sequential):
            kernel = branch[0].weight
            running_mean = branch[1].running_mean
            running_var = branch[1].running_var
            gamma = branch[1].weight
            beta = branch[1].bias
            eps = branch[1].eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, "id_tensor"):
                input_dim = self.in_channels // self.groups
                kernel_value = torch.zeros(
                    (self.in_channels, input_dim, 3, 3), dtype=torch.float32
                )
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = kernel_value
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        assert running_var is not None
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1).to(kernel.device)
        return kernel * t, beta - running_mean * gamma / std


class BlockRepeater(nn.Module):
    def __init__(
        self,
        block: type[nn.Module],
        in_channels: int,
        out_channels: int,
        n_blocks: int = 1,
    ):
        """Module which repeats the block n times. First block accepts
        in_channels and outputs out_channels while subsequent blocks.

        @type block: L{nn.Module}
        @param block: Block to repeat.
        @type in_channels: int
        @param in_channels: Number of input channels.
        @type out_channels: int
        @param out_channels: Number of output channels.
        @type n_blocks: int
        @param n_blocks: Number of blocks to repeat. Defaults to C{1}.
        """
        super().__init__()

        self.blocks = nn.ModuleList()
        self.blocks.append(
            block(in_channels=in_channels, out_channels=out_channels)
        )
        for _ in range(n_blocks - 1):
            self.blocks.append(
                block(in_channels=out_channels, out_channels=out_channels)
            )

    def forward(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class CSPStackRepBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_blocks: int = 1,
        e: float = 0.5,
    ):
        super().__init__()
        """Module composed of three 1x1 conv layers and a stack of sub-
        blocks consisting of two RepVGG blocks with a residual
        connection.

        @type in_channels: int
        @param in_channels: Number of input channels.
        @type out_channels: int
        @param out_channels: Number of output channels.
        @type n_blocks: int
        @param n_blocks: Number of blocks to repeat. Defaults to C{1}.
        @type e: float
        @param e: Factor for number of intermediate channels. Defaults
            to C{0.5}.
        """
        intermediate_channels = int(out_channels * e)
        self.conv_1 = ConvModule(
            in_channels=in_channels,
            out_channels=intermediate_channels,
            kernel_size=1,
            padding=autopad(1, None),
        )
        self.conv_2 = ConvModule(
            in_channels=in_channels,
            out_channels=intermediate_channels,
            kernel_size=1,
            padding=autopad(1, None),
        )
        self.conv_3 = ConvModule(
            in_channels=intermediate_channels * 2,
            out_channels=out_channels,
            kernel_size=1,
            padding=autopad(1, None),
        )
        self.rep_stack = BlockRepeater(
            block=BottleRep,
            in_channels=intermediate_channels,
            out_channels=intermediate_channels,
            n_blocks=n_blocks // 2,
        )

    def forward(self, x: Tensor) -> Tensor:
        out_1 = self.conv_1(x)
        out_1 = self.rep_stack(out_1)
        out_2 = self.conv_2(x)
        out = torch.cat((out_1, out_2), dim=1)
        return self.conv_3(out)


class BottleRep(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        block: type[nn.Module] = RepVGGBlock,
        weight: bool = True,
    ):
        super().__init__()
        """RepVGG bottleneck module.

        @type in_channels: int
        @param in_channels: Number of input channels.
        @type out_channels: int
        @param out_channels: Number of output channels.
        @type block: L{nn.Module}
        @param block: Block to use. Defaults to C{RepVGGBlock}.
        @type weight: bool
        @param weight: If using learnable or static shortcut weight.
            Defaults to C{True}.
        """
        self.conv_1 = block(in_channels=in_channels, out_channels=out_channels)
        self.conv_2 = block(
            in_channels=out_channels, out_channels=out_channels
        )
        self.shortcut = in_channels == out_channels
        self.alpha = nn.Parameter(torch.ones(1)) if weight else 1.0

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv_1(x)
        out = self.conv_2(out)
        return out + self.alpha * x if self.shortcut else out


class SpatialPyramidPoolingBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int = 5
    ):
        """Spatial Pyramid Pooling block with ReLU activation on three
        different scales.

        @type in_channels: int
        @param in_channels: Number of input channels.
        @type out_channels: int
        @param out_channels: Number of output channels.
        @type kernel_size: int
        @param kernel_size: Kernel size. Defaults to C{5}.
        """
        super().__init__()

        intermediate_channels = in_channels // 2  # hidden channels
        self.conv1 = ConvModule(in_channels, intermediate_channels, 1, 1)
        self.conv2 = ConvModule(intermediate_channels * 4, out_channels, 1, 1)
        self.max_pool = nn.MaxPool2d(
            kernel_size=kernel_size, stride=1, padding=kernel_size // 2
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        # apply max-pooling at three different scales
        y1 = self.max_pool(x)
        y2 = self.max_pool(y1)
        y3 = self.max_pool(y2)

        x = torch.cat([x, y1, y2, y3], dim=1)
        x = self.conv2(x)
        return x


class AttentionRefinmentBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        """Attention Refinment block adapted from
        U{https://github.com/taveraantonio/BiseNetv1}.

        @type in_channels: int
        @param in_channels: Number of input channels.
        @type out_channels: int
        @param out_channels: Number of output channels.
        """
        super().__init__()

        self.conv_3x3 = ConvModule(in_channels, out_channels, 3, 1, 1)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvModule(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=1,
                activation=False,
            ),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_3x3(x)
        attention = self.attention(x)
        out = x * attention
        return out


class FeatureFusionBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, reduction: int = 1
    ):
        """Feature Fusion block adapted from: U{https://github.com/taveraantonio/BiseNetv1}.

        @type in_channels: int
        @param in_channels: Number of input channels.
        @type out_channels: int
        @param out_channels: Number of output channels.
        @type reduction: int
        @param reduction: Reduction factor. Defaults to C{1}.
        """
        super().__init__()

        self.conv_1x1 = ConvModule(in_channels, out_channels, 1, 1, 0)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvModule(
                in_channels=out_channels,
                out_channels=out_channels // reduction,
                kernel_size=1,
            ),
            ConvModule(
                in_channels=out_channels,
                out_channels=out_channels // reduction,
                kernel_size=1,
                activation=False,
            ),
            nn.Sigmoid(),
        )

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        fusion = torch.cat([x1, x2], dim=1)
        x = self.conv_1x1(fusion)
        attention = self.attention(x)
        out = x + x * attention
        return out


T = TypeVar("T", int, tuple[int, ...])


def autopad(kernel_size: T, padding: T | None = None) -> T:
    """Compute padding based on kernel size.

    @type kernel_size: int | tuple[int, ...]
    @param kernel_size: Kernel size.
    @type padding: int | tuple[int, ...] | None
    @param padding: Padding. Defaults to None.

    @rtype: int | tuple[int, ...]
    @return: Computed padding. The output type is the same as the type of the
        C{kernel_size}.
    """
    if padding is not None:
        return padding
    if isinstance(kernel_size, int):
        return kernel_size // 2
    return tuple(x // 2 for x in kernel_size)


class BasicResNetBlock(nn.Module):
    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        expansion: int = 1,
        final_relu: bool = True,
        droppath_prob: float = 0.0,
    ):
        """A basic residual block for ResNet.

        @type in_planes: int
        @param in_planes: Number of input channels.
        @type planes: int
        @param planes: Number of output channels.
        @type stride: int
        @param stride: Stride for the convolutional layers. Defaults to 1.
        @type expansion: int
        @param expansion: Expansion factor for the output channels. Defaults to 1.
        @type final_relu: bool
        @param final_relu: Whether to apply a ReLU activation after the residual
            addition. Defaults to True.
        @type droppath_prob: float
        @param droppath_prob: Drop path probability for stochastic depth. Defaults to
            0.0.
        """
        super().__init__()
        self.expansion = expansion
        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.final_relu = final_relu

        self.drop_path = DropPath(drop_prob=droppath_prob)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x: Tensor) -> Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.drop_path(out)
        out += self.shortcut(x)
        if self.final_relu:
            out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        expansion: int = 4,
        final_relu: bool = True,
        droppath_prob: float = 0.0,
    ):
        """A bottleneck block for ResNet.

        @type in_planes: int
        @param in_planes: Number of input channels.
        @type planes: int
        @param planes: Number of intermediate channels.
        @type stride: int
        @param stride: Stride for the second convolutional layer. Defaults to 1.
        @type expansion: int
        @param expansion: Expansion factor for the output channels. Defaults to 4.
        @type final_relu: bool
        @param final_relu: Whether to apply a ReLU activation after the residual
            addition. Defaults to True.
        @type droppath_prob: float
        @param droppath_prob: Drop path probability for stochastic depth. Defaults to
            0.0.
        """
        super().__init__()
        self.expansion = expansion
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.final_relu = final_relu

        self.drop_path = DropPath(drop_prob=droppath_prob)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x: Tensor) -> Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        out = self.drop_path(out)
        out += self.shortcut(x)

        if self.final_relu:
            out = F.relu(out)

        return out


class UpscaleOnline(nn.Module):
    """Upscale tensor to a specified size during the forward pass.

    This class supports cases where the required scale/size is only
    known when the input is received. Only the interpolation mode is set
    in advance.
    """

    def __init__(self, mode: str = "bilinear"):
        """Initialize UpscaleOnline with the interpolation mode.

        @type mode: str
        @param mode: Interpolation mode for resizing. Defaults to
            "bilinear".
        """
        super().__init__()
        self.mode = mode

    def forward(
        self, x: Tensor, output_height: int, output_width: int
    ) -> Tensor:
        """Upscale the input tensor to the specified height and width.

        @type x: Tensor
        @param x: Input tensor to be upscaled.
        @type output_height: int
        @param output_height: Desired height of the output tensor.
        @type output_width: int
        @param output_width: Desired width of the output tensor.
        @return: Upscaled tensor.
        """
        return F.interpolate(
            x, size=[output_height, output_width], mode=self.mode
        )


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample, when applied in the
    main path of residual blocks.

    Intended usage of this block is as follows:

    >>> class ResNetBlock(nn.Module):
    ...   def __init__(self, ..., drop_path_rate: float):
    ...     self.drop_path = DropPath(drop_path_rate)

    ...   def forward(self, x):
    ...     return x + self.drop_path(self.conv_bn_act(x))

    @see: U{Original code (TIMM) <https://github.com/rwightman/pytorch-image-models>}
    @license: U{Apache License 2.0 <https://github.com/huggingface/pytorch-image-models?tab=Apache-2.0-1-ov-file#readme>}
    """

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        """Initializes the DropPath module.

        @type drop_prob: float
        @param drop_prob: Probability of zeroing out individual vectors
            (channel dimension) of each feature map. Defaults to 0.0.
        @type scale_by_keep: bool
        @param scale_by_keep: Whether to scale the output by the keep
            probability. Enabled by default to maintain output mean &
            std in the same range as without DropPath. Defaults to True.
        """
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def drop_path(
        self, x: Tensor, drop_prob: float = 0.0, scale_by_keep: bool = True
    ) -> Tensor:
        """Drop paths (Stochastic Depth) per sample when applied in the
        main path of residual blocks.

        @type x: Tensor
        @param x: Input tensor.
        @type drop_prob: float
        @param drop_prob: Probability of dropping a path. Defaults to
            0.0.
        @type scale_by_keep: bool
        @param scale_by_keep: Whether to scale the output by the keep
            probability. Defaults to True.
        @return: Tensor with dropped paths based on the provided drop
            probability.
        """
        keep_prob = 1 - drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor

    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        return self.drop_path(x, self.drop_prob, self.scale_by_keep)
