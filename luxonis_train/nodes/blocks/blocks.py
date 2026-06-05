from collections.abc import Callable
from typing import Literal, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from typeguard import typechecked
from typing_extensions import override

from .reparametrizable import Reparametrizable
from .utils import ModuleFactory, autopad


class PreciseDecoupledBlock(nn.Module):
    __call__: Callable[[Tensor], tuple[Tensor, Tensor, Tensor]]

    @typechecked
    def __init__(
        self,
        in_channels: int,
        reg_channels: int,
        cls_channels: int,
        n_classes: int,
        reg_max: int,
    ):
        super().__init__()
        self.classification_branch = nn.Sequential(
            ConvBlock(
                in_channels,
                cls_channels,
                kernel_size=3,
                padding=1,
                activation=nn.SiLU(),
            ),
            ConvBlock(
                cls_channels,
                cls_channels,
                kernel_size=3,
                padding=1,
                activation=nn.SiLU(),
            ),
            nn.Conv2d(cls_channels, n_classes, kernel_size=1),
        )
        self.regression_branch = nn.Sequential(
            ConvBlock(
                in_channels,
                reg_channels,
                kernel_size=3,
                padding=1,
                activation=nn.SiLU(),
            ),
            ConvBlock(
                reg_channels,
                reg_channels,
                kernel_size=3,
                padding=1,
                activation=nn.SiLU(),
            ),
            nn.Conv2d(reg_channels, 4 * reg_max, kernel_size=1),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        regressions = self.regression_branch(x)
        classes = self.classification_branch(x)
        features = torch.cat([regressions, classes], dim=1)
        return features, classes, regressions


class EfficientDecoupledBlock(nn.Module):
    __call__: Callable[[Tensor], tuple[Tensor, Tensor, Tensor]]

    @typechecked
    def __init__(self, in_channels: int, n_classes: int):
        """Efficient decoupled block for class and regression
        predictions.

        Args:
            in_channels (int): Number of input channels.
            n_classes (int): Number of classes.

        """
        super().__init__()

        self.decoder = ConvBlock(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            activation=nn.SiLU(),
        )

        self.class_branch = nn.Sequential(
            ConvBlock(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                activation=nn.SiLU(),
            ),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=n_classes,
                kernel_size=1,
            ),
        )
        self.regression_branch = nn.Sequential(
            ConvBlock(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                activation=nn.SiLU(),
            ),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=4,
                kernel_size=1,
            ),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        features = self.decoder(x)

        classes = self.class_branch(features)
        regressions = self.regression_branch(features)

        return features, classes, regressions


class SegProto(nn.Sequential):
    @typechecked
    def __init__(
        self, in_channels: int, mid_channels: int = 256, out_channels: int = 32
    ):
        """Initialize the segmentation prototype generator.

        Args:
            in_channels (int): Number of input channels.
            mid_channels (int): Number of intermediate channels. Defaults to 256.
            out_channels (int): Number of output channels. Defaults to 32.

        """
        super().__init__(
            ConvBlock(
                in_channels=in_channels,
                out_channels=mid_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                activation=nn.SiLU(),
            ),
            nn.ConvTranspose2d(
                in_channels=mid_channels,
                out_channels=mid_channels,
                kernel_size=2,
                stride=2,
                bias=True,
            ),
            ConvBlock(
                in_channels=mid_channels,
                out_channels=mid_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                activation=nn.SiLU(),
            ),
            ConvBlock(
                in_channels=mid_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                activation=nn.SiLU(),
            ),
        )


class DFL(nn.Module):
    """Process distribution focal loss logits.

    The DFL (Distribution Focal Loss) module processes input tensors by
    applying softmax over a specified dimension and projecting the
    resulting tensor to produce output logits.

    """

    @typechecked
    def __init__(self, reg_max: int = 16):
        """Initialize the DFL module.

        Args:
            reg_max (int): Maximum number of regression outputs. Defaults to 16.

        """
        super().__init__()
        self.conv = nn.Conv2d(reg_max, 1, kernel_size=1, bias=False)
        self.conv.weight.data.copy_(
            torch.arange(reg_max, dtype=torch.float32).view(1, reg_max, 1, 1)
        )
        self.conv.requires_grad_(False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: Tensor) -> Tensor:
        n, _, h, w = x.size()
        x = x.view(n, 4, -1, h * w).permute(0, 2, 1, 3)
        x = self.softmax(x)
        return self.conv(x)[:, 0].view(n, 4, h, w)


class ConvBlock(nn.Module):
    @typechecked
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
        activation: Callable[[Tensor], Tensor] | None | bool = True,
        use_norm: bool = True,
        norm_momentum: float = 0.1,
    ):
        """Conv2d + Optional BN + Activation.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Kernel size.
            stride (int): Stride. Defaults to 1.
            padding (int | str): Padding. Defaults to 0.
            dilation (int): Dilation. Defaults to 1.
            groups (int): Groups. Defaults to 1.
            bias (bool): Whether to use bias. Defaults to False.
            activation (``nn.Module | None | bool``): Activation function. Defaults to ``nn.ReLu`` if not explicitly set to ``None`` or ``False``.
            use_norm (bool): Whether to use batch normalization. Defaults to True.
            norm_momentum (float): Momentum for batch normalization. Defaults to 0.1.

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
            self.bn = nn.BatchNorm2d(out_channels, momentum=norm_momentum)

        if activation is True:
            self.activation = nn.ReLU()
        elif not activation:
            self.activation = nn.Identity()
        else:
            self.activation = activation

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        return self.activation(x)


class SqueezeExciteBlock(nn.Sequential):
    @typechecked
    def __init__(
        self,
        in_channels: int,
        intermediate_channels: int,
        hard_sigmoid: bool = False,
        activation: nn.Module | None = None,
    ):
        """Squeeze and Excite block.

        Adapted from `Squeeze-and-Excitation Networks <https://arxiv.org/pdf/1709.01507.pdf>`_.
        Code adapted from `https://github.com/apple/ml-mobileone/blob/main/mobileone.py <https://github.com/apple/ml-mobileone/blob/main/mobileone.py>`_.

        Args:
            in_channels (int): Number of input channels.
            intermediate_channels (int): Number of intermediate channels.
            hard_sigmoid (bool): Whether to use hard sigmoid function. Defaults to False.
            activation (``nn.Module | None``): Activation function. Defaults to ``nn.ReLU``.

        """
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=intermediate_channels,
                kernel_size=1,
                bias=True,
            ),
            activation or nn.ReLU(),
            nn.Conv2d(
                in_channels=intermediate_channels,
                out_channels=in_channels,
                kernel_size=1,
                bias=True,
            ),
            nn.Hardsigmoid() if hard_sigmoid else nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x * super().forward(x)


# TODO: Maybe a better name?
class GeneralReparametrizableBlock(Reparametrizable):
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
        n_branches: int = 1,
        # TODO: Maybe a better name?
        refine_block: nn.Module | Literal["se"] | None = None,
        use_scale_layer: bool = True,
        scale_layer_padding: int | tuple[int, int] | None = None,
        activation: nn.Module | None | bool = True,
    ):
        """General reparametrizable block with train and deploy states.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Kernel size. Defaults to ``3``.
            stride (int): Stride. Defaults to ``1``.
            padding (int): Padding. Defaults to ``1``.
            groups (int): Groups. Defaults to ``1``.
            n_branches (int): Number of convolutional branches. During reparametrization, the branches are fused to a single convolutional layer. Defaults to ``1``.
            refine_block (``nn.Module | Literal["se"] | None``): A block to refine the output. Placed after the convolutional branches and before the activation function. Can be one of the following: - torch module - string ``"se"`` which will use `SqueezeExciteBlock` - None for no operation Defaults to ``None``.
            use_scale_layer (bool): Whether to add a 1x1 scale branch. Defaults to ``True``.
            scale_layer_padding (int | tuple[int, int] | None): Padding for the scale branch. Defaults to None.
            activation (``nn.Module | None | bool``): Activation function. By default ``nn.ReLU``. If ``False`` or ``None`` then no activation.

        See Also:
            `https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py <https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py>`_.

        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.groups = groups

        self.skip_layer: nn.BatchNorm2d | None = None
        if out_channels == in_channels and stride in (1, (1, 1)):
            self.skip_layer = nn.BatchNorm2d(in_channels)

        self.scale_layer: ConvBlock | None = None

        if use_scale_layer:
            padding_scale = scale_layer_padding or padding - kernel_size // 2
            self.scale_layer = ConvBlock(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=1,
                stride=stride,
                padding=padding_scale,
                groups=self.groups,
                activation=False,
            )

        branches = [
            ConvBlock(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=self.groups,
                activation=False,
            )
            for _ in range(n_branches)
        ]

        if refine_block == "se":
            self.refine_block = SqueezeExciteBlock(
                in_channels=out_channels,
                intermediate_channels=out_channels // 16,
            )
        else:
            self.refine_block = refine_block or nn.Identity()

        if activation is True:
            self.activation = nn.ReLU()
        elif not activation:
            self.activation = nn.Identity()
        else:
            self.activation = activation or nn.ReLU()

        self.branches = cast(list[ConvBlock], nn.ModuleList(branches))
        self.fused_branch: nn.Conv2d | None = None

    def forward(self, x: Tensor) -> Tensor:
        if self.fused_branch is None:
            out = 0

            if self.skip_layer is not None:
                out += self.skip_layer(x)

            for branch in self.branches:
                out += branch(x)

            if self.scale_layer is not None:
                out += self.scale_layer(x)
        else:
            out = self.fused_branch(x)

        return self.activation(self.refine_block(out))

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @override
    def reparametrize(self) -> None:
        """Fuse training-time branches into a single convolution branch.

        The method creates ``fused_branch`` from the dense, scale, and
        skip branches and switches subsequent forward passes to the
        fused branch. It raises ``RuntimeError`` if the block has
        already been reparametrized.

        """
        if self.fused_branch is not None:
            raise RuntimeError(f"{self.name} is already reparametrized")

        kernel, bias = self._fuse_parameters()
        fused_branch = nn.Conv2d(
            in_channels=self.branches[0].in_channels,
            out_channels=self.branches[0].out_channels,
            kernel_size=self.branches[0].kernel_size,
            stride=self.branches[0].stride,
            padding=self.branches[0].padding,
            dilation=self.branches[0].dilation,
            groups=self.branches[0].groups,
            bias=True,
        )
        fused_branch.weight.data = kernel
        assert fused_branch.bias is not None
        fused_branch.bias.data = bias

        self.fused_branch = fused_branch

    @override
    def restore(self) -> None:
        if self.fused_branch is None:
            raise RuntimeError(
                f"Cannot restore '{self.name}' "
                "that has not yet been reparametrized."
            )

        # Not sure if this is necessary
        for param in self.fused_branch.parameters():
            param.detach_()

        del self.fused_branch
        self.fused_branch = None

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
            kernel += F.pad(kernel_scale, [pad, pad, pad, pad])
            bias += bias_scale

        if self.skip_layer is not None:
            kernel_identity, bias_identity = self._fuse_batch_norm(
                self.skip_layer
            )
            kernel += kernel_identity
            bias += bias_identity

        return kernel, bias

    def _fuse_conv(self, module: ConvBlock) -> tuple[Tensor, Tensor]:
        kernel = module.conv.weight
        assert module.bn is not None
        running_mean = module.bn.running_mean
        running_var = module.bn.running_var
        gamma = module.bn.weight
        beta = module.bn.bias
        eps = module.bn.eps
        return self._postprocess_fused(
            running_var, running_mean, gamma, beta, kernel, eps
        )

    def _fuse_batch_norm(
        self, module: nn.BatchNorm2d
    ) -> tuple[Tensor, Tensor]:
        input_dim = self.in_channels // self.groups
        kernel = torch.zeros(
            (self.in_channels, input_dim, self.kernel_size, self.kernel_size),
            dtype=module.weight.dtype,
            device=module.weight.device,
        )
        for i in range(self.in_channels):
            kernel[
                i, i % input_dim, self.kernel_size // 2, self.kernel_size // 2
            ] = 1

        running_mean = module.running_mean
        running_var = module.running_var
        gamma = module.weight
        beta = module.bias
        eps = module.eps
        return self._postprocess_fused(
            running_var, running_mean, gamma, beta, kernel, eps
        )

    def _postprocess_fused(
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


class BlockRepeater(nn.Sequential):
    """Module which repeats a given block n times.

    If the block has an ``out_channels`` and ``in_channels`` argument,
    the ``in_channels`` of the next block will be set to the
    ``out_channels`` of the previous block. This allows for repeating
    blocks which change the number of channels.

    """

    @typechecked
    def __init__(
        self, module: Callable[..., nn.Module], /, *, n_repeats: int, **kwargs
    ):
        """Initialize the repeated block stack.

        Args:
            module (``Callable[..., nn.Module]``): Module factory to repeat.
            n_repeats (int): Number of blocks to repeat. Defaults to ``1``.
            **kwargs (``Any``): Keyword arguments forwarded to ``module``.

        """
        blocks = [module(**kwargs)]

        if "out_channels" in kwargs:
            kwargs["in_channels"] = kwargs["out_channels"]

        for _ in range(n_repeats - 1):
            blocks.append(module(**kwargs))

        super().__init__(*blocks)


class CSPStackRepBlock(nn.Module):
    """Stack RepVGG-style residual blocks inside a CSP block."""

    @typechecked
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_blocks: int = 1,
        e: float = 0.5,
    ):
        """Initialize the CSP stack.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            n_blocks (int): Number of blocks to repeat. Defaults to ``1``.
            e (float): Factor for number of intermediate channels. Defaults to ``0.5``.

        """
        super().__init__()
        intermediate_channels = int(out_channels * e)
        self.conv_1 = ConvBlock(
            in_channels=in_channels,
            out_channels=intermediate_channels,
            kernel_size=1,
            padding=autopad(1, None),
        )
        self.rep_stack = BlockRepeater(
            BottleRep,
            in_channels=intermediate_channels,
            out_channels=intermediate_channels,
            n_repeats=n_blocks // 2,
        )
        self.conv_2 = ConvBlock(
            in_channels=in_channels,
            out_channels=intermediate_channels,
            kernel_size=1,
            padding=autopad(1, None),
        )
        self.conv_3 = ConvBlock(
            in_channels=intermediate_channels * 2,
            out_channels=out_channels,
            kernel_size=1,
            padding=autopad(1, None),
        )

    def forward(self, x: Tensor) -> Tensor:
        out_1 = self.conv_1(x)
        out_1 = self.rep_stack(out_1)
        out_2 = self.conv_2(x)
        out = torch.cat([out_1, out_2], dim=1)
        return self.conv_3(out)


class BottleRep(nn.Module):
    @typechecked
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        module: ModuleFactory = GeneralReparametrizableBlock,
        weight: bool = True,
        **kwargs,
    ):
        """RepVGG bottleneck module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            module (ModuleFactory): Block factory to use. Defaults to `GeneralReparametrizableBlock`.
            weight (bool): If using learnable or static shortcut weight. Defaults to ``True``.
            **kwargs (``Any``): Keyword arguments forwarded to ``module``.

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
        """Spatial Pyramid Pooling block with three ReLU-activated
        scales.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Kernel size. Defaults to ``5``.

        """
        super().__init__()

        intermediate_channels = in_channels // 2  # hidden channels
        self.conv1 = ConvBlock(in_channels, intermediate_channels, 1, 1)
        self.conv2 = ConvBlock(intermediate_channels * 4, out_channels, 1, 1)
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
        return self.conv2(x)


class AttentionRefinmentBlock(nn.Module):
    @typechecked
    def __init__(self, in_channels: int, out_channels: int):
        """Attention Refinment block.

        Adapted from `https://github.com/taveraantonio/BiseNetv1 <https://github.com/taveraantonio/BiseNetv1>`_.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.

        """
        super().__init__()

        self.conv = ConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBlock(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=1,
                activation=nn.Sigmoid(),
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        attention = self.attention(x)
        return x * attention


class FeatureFusionBlock(nn.Module):
    @typechecked
    def __init__(
        self, in_channels: int, out_channels: int, reduction: int = 1
    ):
        """Feature Fusion block adapted from: `https://github.com/taveraantonio/BiseNetv1 <https://github.com/taveraantonio/BiseNetv1>`_.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            reduction (int): Reduction factor. Defaults to ``1``.

        """
        super().__init__()

        self.conv_1x1 = ConvBlock(in_channels, out_channels, 1, 1, 0)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBlock(
                in_channels=out_channels,
                out_channels=out_channels // reduction,
                kernel_size=1,
            ),
            ConvBlock(
                in_channels=out_channels,
                out_channels=out_channels // reduction,
                kernel_size=1,
                activation=None,
            ),
            nn.Sigmoid(),
        )

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        fusion = torch.cat([x1, x2], dim=1)
        x = self.conv_1x1(fusion)
        attention = self.attention(x)
        return x + x * attention


class UpscaleOnline(nn.Module):
    """Upscale tensor to a specified size during the forward pass.

    This class supports cases where the required scale or size is only
    known when the input is received. Only the interpolation mode is set
    in advance.

    Args:
        mode (str): Interpolation mode for resizing. Defaults to ``"bilinear"``.

    """

    @typechecked
    def __init__(self, mode: str = "bilinear"):
        super().__init__()
        self.mode = mode

    def forward(
        self, x: Tensor, output_height: int, output_width: int
    ) -> Tensor:
        """Upscale the input tensor to the specified height and width.

        Args:
            x (``Tensor``): Input tensor to be upscaled.
            output_height (int): Desired height of the output tensor.
            output_width (int): Desired width of the output tensor.

        Returns:
            ``Tensor``: Upscaled tensor.

        """
        return F.interpolate(
            x, size=[output_height, output_width], mode=self.mode
        )


class DropPath(nn.Module):
    """Drop paths per sample in the main path of residual blocks.

    Args:
        drop_prob (float): Probability of zeroing out each sample path.
            Defaults to ``0.0``.
        scale_by_keep (bool): Whether to scale the output by the keep
            probability. Defaults to ``True``.

    Notes:
        License: `Apache License 2.0 <https://github.com/huggingface/pytorch-image-models?tab=Apache-2.0-1-ov-file#readme>`_.

    See Also:
        `Original code (TIMM) <https://github.com/rwightman/pytorch-image-models>`_.

    """

    @typechecked
    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def drop_path(self, x: Tensor) -> Tensor:
        """Drop paths per sample when training.

        Args:
            x (``Tensor``): Input tensor.

        Returns:
            ``Tensor``: ``Tensor`` with paths dropped according to ``drop_prob``.

        """
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor

    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        return self.drop_path(x)


class ConvStack(BlockRepeater):
    def __init__(
        self, in_channels: int, out_channels: int, *, n_repeats: int = 2
    ):
        """Stack convolution blocks.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            n_repeats (int): Number of ``ConvBlock`` modules to stack.

        """
        super().__init__(
            ConvBlock,
            n_repeats=n_repeats,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
        )
