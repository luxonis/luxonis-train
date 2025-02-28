import math
from collections.abc import Callable
from typing import Generic, Literal, TypeVar, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from typeguard import typechecked
from typing_extensions import override

from luxonis_train.nodes.activations import HSigmoid

from .reparametrizable import Reparametrizable


class EfficientDecoupledBlock(nn.Module):
    @typechecked
    def __init__(
        self, n_classes: int, in_channels: int, prior_probability: float = 1e-2
    ):
        """Efficient Decoupled block used for class and regression
        predictions.

        @type n_classes: int
        @param n_classes: Number of classes.
        @type in_channels: int
        @param in_channels: Number of input channels.
        @type prior_probability: float
        @param prior_probability: ???
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

        self._initialize_weights_and_biases(prior_probability)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        features = self.decoder(x)

        classes = self.class_branch(features)
        regressions = self.regression_branch(features)

        return features, classes, regressions

    def _initialize_weights_and_biases(self, p: float) -> None:
        data = [
            (self.class_branch[-1], -math.log((1 - p) / p)),
            (self.regression_branch[-1], 1.0),
        ]
        for module, fill_value in data:
            assert isinstance(module, nn.Conv2d)
            assert module.bias is not None

            b = module.bias.view(-1)
            b.data.fill_(fill_value)
            module.bias = nn.Parameter(b, requires_grad=True)

            w = module.weight
            w.data.fill_(0.0)
            module.weight = nn.Parameter(w, requires_grad=True)


class SegProto(nn.Module):
    @typechecked
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
    @typechecked
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


class ConvModule(nn.Module):
    @typechecked
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = False,
        activation: nn.Module | None | Literal[False] = None,
        use_norm: bool = True,
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
        @type padding: int
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
        @param use_norm: Whether to use batch normalization. Defaults to
            True.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )

        self.bn: nn.BatchNorm2d | None = None
        if use_norm:
            self.bn = nn.BatchNorm2d(out_channels)

        self.activation: nn.Module | None = None
        if activation is not False:
            self.activation = activation or nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class UpBlock(nn.Sequential):
    @typechecked
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
        else:
            layers.append(
                nn.Upsample(
                    scale_factor=stride,
                    mode=inter_mode,
                    align_corners=align_corners,
                )
            )
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))

        layers.append(
            ConvModule(out_channels, out_channels, kernel_size=3, padding=1)
        )

        super().__init__(*layers)


class SqueezeExciteBlock(nn.Module):
    @typechecked
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
        self.pool = nn.AdaptiveAvgPool2d(1)
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


RefB = TypeVar("RefB", bound=nn.Module)


class GeneralReparametrizableBlock(
    nn.Module,
    Reparametrizable,
    Generic[RefB],
):
    __call__: Callable[[Tensor], Tensor]

    @typechecked
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        groups: int = 1,
        num_branches: int = 1,
        refine_block: RefB | Literal["se"] | None = None,
        activation: nn.Module | None | Literal[False] = None,
    ):
        """GeneralReparametrizableBlock is a basic rep-style block,
        including training and deploy status.

        @see: U{https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py}.

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
        @type groups: int
        @param groups: Groups. Defaults to C{1}.
        @type num_branches: int
        @param num_branches: Number of convolutional branches.
            During reparametrization, the branches are fused to a single
            convolutional layer. Defaults to C{1}.
        @type refine_block: nn.Module | Literal["se"] | None
        @param refine_block: A block to refine the output.
            Placed after the convolutional branches and before the
            activation function.
            Can be one of the following:
              - torch module
              - string `"se"` which will use L{SqueezeExciteBlock}
            Defaults to C{None}.
        @type activation: nn.Module | None | Literal[False]
        @param activation: Activation function. If C{None} then C{nn.ReLU}.
            If C{False} then no activation. Defaults to C{nn.ReLU}.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.groups = groups

        self._reparametrized = False

        self.skip_layer: nn.BatchNorm2d | None = None
        if out_channels == in_channels and stride == 1:
            self.skip_layer = nn.BatchNorm2d(in_channels)

        self.scale_layer: ConvModule | None = None
        padding_scale = padding - kernel_size // 2
        if padding_scale > 0:
            self.scale_layer = ConvModule(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=1,
                stride=stride,
                padding=padding_scale,
                groups=self.groups,
                activation=False,
            )

        branches = [
            ConvModule(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=self.groups,
                activation=False,
            )
            for _ in range(num_branches)
        ]

        if refine_block == "se":
            self.refine_block = SqueezeExciteBlock(
                in_channels=out_channels,
                intermediate_channels=out_channels // 16,
            )
        else:
            self.refine_block = refine_block or nn.Identity()

        if activation is False:
            self.activation = nn.Identity()
        else:
            self.activation = activation or nn.ReLU()

        self.branches = cast(list[ConvModule], nn.ModuleList(branches))

    def forward(self, x: Tensor) -> Tensor:
        out = 0
        if self.skip_layer is not None:
            out += self.skip_layer(x)

        if self.scale_layer is not None:
            out += self.scale_layer(x)
        for branch in self.branches:
            out += branch(x)

        return self.activation(self.refine_block(out))

    @override
    def reparametrize(self) -> None:
        if self._reparametrized:
            raise RuntimeError(
                f"{self.__class__.__name__} is already reparametrized"
            )

        kernel, bias = self._fuse_parameters()
        rep_dense_block = nn.Conv2d(
            in_channels=self.branches[0].in_channels,
            out_channels=self.branches[0].out_channels,
            kernel_size=self.branches[0].kernel_size,
            stride=self.branches[0].stride,
            padding=self.branches[0].padding,
            dilation=self.branches[0].dilation,
            groups=self.branches[0].groups,
            bias=True,
        )
        rep_dense_block.weight.data = kernel
        if rep_dense_block.bias is not None:
            rep_dense_block.bias.data = bias

        # NOTE: Not sure if the explicit `detach_` and
        # deletions are necessary. Some of the reference
        # implementations include it and some don't.
        for para in self.parameters():
            para.detach_()

        del self.branches
        del self.scale_layer
        del self.skip_layer

        self.branches = cast(
            list[ConvModule], nn.ModuleList([rep_dense_block])
        )
        self.scale_layer = None
        self.skip_layer = None

        self._reparametrized = True

    def _fuse_parameters(self) -> tuple[Tensor, Tensor]:
        kernel = torch.tensor(0)
        bias = torch.tensor(0)

        for dense_block in self.branches:
            kernel_dense, bias_dense = self._fuse_conv(dense_block)
            kernel = kernel_dense + kernel
            bias = bias_dense + bias

        if self.scale_layer is not None:
            kernel_scale, bias_scale = self._fuse_conv(self.scale_layer)
            pad = self.kernel_size // 2
            kernel += torch.nn.functional.pad(
                kernel_scale, [pad, pad, pad, pad]
            )
            bias += bias_scale

        if self.skip_layer is not None:
            kernel_identity, bias_identity = self._fuse_batch_norm(
                self.skip_layer
            )
            kernel += kernel_identity
            bias += bias_identity

        return kernel, bias

    def _fuse_conv(self, module: ConvModule) -> tuple[Tensor, Tensor]:
        kernel = module.conv.weight
        assert module.bn is not None
        running_mean = module.bn.running_mean
        running_var = module.bn.running_var
        gamma = module.bn.weight
        beta = module.bn.bias
        eps = module.bn.eps
        return self._postprocess_fusion(
            running_var,
            running_mean,
            gamma,
            beta,
            kernel,
            eps,
        )

    def _fuse_batch_norm(
        self, module: nn.BatchNorm2d
    ) -> tuple[Tensor, Tensor]:
        input_dim = self.in_channels // self.groups
        kernel = torch.zeros(
            (
                self.in_channels,
                input_dim,
                self.kernel_size,
                self.kernel_size,
            ),
            dtype=module.weight.dtype,
            device=module.weight.device,
        )
        for i in range(self.in_channels):
            kernel[
                i,
                i % input_dim,
                self.kernel_size // 2,
                self.kernel_size // 2,
            ] = 1

        running_mean = module.running_mean
        running_var = module.running_var
        gamma = module.weight
        beta = module.bias
        eps = module.eps
        return self._postprocess_fusion(
            running_var, running_mean, gamma, beta, kernel, eps
        )

    def _postprocess_fusion(
        self,
        running_var: Tensor | None,
        running_mean: Tensor | None,
        gamma: Tensor,
        beta: Tensor,
        kernel: Tensor,
        eps: float,
    ) -> tuple[Tensor, Tensor]:
        if running_var is None or running_mean is None:
            raise ValueError(
                "Running variance and mean must be "
                "provided for reparametrization."
            )
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1).to(kernel.device)
        return kernel * t, beta - running_mean * gamma / std


class ModuleRepeater(nn.Sequential):
    @typechecked
    def __init__(
        self,
        module: Callable[..., nn.Module],
        n_repeats: int,
        **kwargs,
    ):
        """Module which repeats the block n times. First block accepts
        in_channels and outputs out_channels while subsequent blocks
        accept out_channels and output out_channels.

        @type module: C{type[nn.Module]}
        @param module: Module to repeat.
        @type n_repeats: int
        @param n_repeats: Number of blocks to repeat. Defaults to C{1}.
        @param kwargs: Additional keyword arguments to be passed to the
            module.
        """
        blocks = [module(**kwargs)]

        if "out_channels" in kwargs:
            kwargs["in_channels"] = kwargs["out_channels"]

        for _ in range(n_repeats - 1):
            blocks.append(module(**kwargs))

        super().__init__(*blocks)


class CSPStackRepBlock(nn.Module):
    @typechecked
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_blocks: int = 1,
        e: float = 0.5,
    ):
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
        super().__init__()
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
        self.rep_stack = ModuleRepeater(
            module=BottleRep,
            in_channels=intermediate_channels,
            out_channels=intermediate_channels,
            n_repeats=n_blocks // 2,
        )

    def forward(self, x: Tensor) -> Tensor:
        out_1 = self.conv_1(x)
        out_1 = self.rep_stack(out_1)
        out_2 = self.conv_2(x)
        out = torch.cat((out_1, out_2), dim=1)
        return self.conv_3(out)


class BottleRep(nn.Module):
    @typechecked
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        module: Callable[..., nn.Module] = GeneralReparametrizableBlock,
        weight: bool = True,
        **kwargs,
    ):
        """RepVGG bottleneck module.

        @type block: L{type[nn.Module]}
        @param block: Block to use. Defaults to
            L{GeneralReparametrizableBlock}.
        @type in_channels: int
        @param in_channels: Number of input channels.
        @type out_channels: int
        @param out_channels: Number of output channels.
        @type weight: bool
        @param weight: If using learnable or static shortcut weight.
            Defaults to C{True}.
        @param kwargs: Additional keyword arguments to be passed to the
            module.
        """
        super().__init__()
        self.conv_1 = module(
            in_channels=in_channels, out_channels=out_channels, **kwargs
        )
        self.conv_2 = module(
            in_channels=out_channels, out_channels=out_channels, **kwargs
        )
        self.shortcut = in_channels == out_channels
        self.alpha = nn.Parameter(torch.ones(1)) if weight else 1.0

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv_1(x)
        out = self.conv_2(out)
        return out + self.alpha * x if self.shortcut else out


class SpatialPyramidPoolingBlock(nn.Module):
    @typechecked
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
    @typechecked
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
    @typechecked
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


class ResNetBlock(nn.Module):
    @typechecked
    def __init__(
        self,
        in_channels: int,
        planes: int,
        stride: int = 1,
        expansion: int = 1,
        final_relu: bool = True,
        droppath_prob: float = 0.0,
    ):
        """A basic residual block for ResNet.

        @type in_channels: int
        @param in_channels: Number of input channels.
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
            in_channels,
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
        if stride != 1 or in_channels != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
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


class ResNetBottleneck(nn.Module):
    @typechecked
    def __init__(
        self,
        in_channels: int,
        planes: int,
        stride: int = 1,
        expansion: int = 4,
        final_relu: bool = True,
        droppath_prob: float = 0.0,
    ):
        """A bottleneck block for ResNet.

        @type in_channels: int
        @param in_channels: Number of input channels.
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
        self.conv1 = nn.Conv2d(
            in_channels,
            planes,
            kernel_size=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.final_relu = final_relu

        self.drop_path = DropPath(drop_prob=droppath_prob)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
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

    @type mode: str
    @param mode: Interpolation mode for resizing. Defaults to
        "bilinear".
    """

    @typechecked
    def __init__(self, mode: str = "bilinear"):
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

    @type drop_prob: float
    @param drop_prob: Probability of zeroing out individual vectors
        (channel dimension) of each feature map. Defaults to 0.0.
    @type scale_by_keep: bool
    @param scale_by_keep: Whether to scale the output by the keep
        probability. Enabled by default to maintain output mean &
        std in the same range as without DropPath. Defaults to True.
    """

    @typechecked
    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
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
