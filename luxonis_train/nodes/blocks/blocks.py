"""The layer blocks that more than one node uses."""

from collections.abc import Callable
from typing import Literal, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from typeguard import typechecked
from typing_extensions import override

from .reparameterizable import Reparameterizable
from .utils import ModuleFactory, autopad


class PreciseDecoupledBlock(nn.Module):
    """Decoupled head of `PrecisionBBoxHead` for one scale.

    The block runs two parallel branches on the same feature map. Each
    branch is two ``3x3`` `ConvBlock` layers with batch norm and SiLU,
    followed by a ``1x1`` convolution. The classification branch ends
    with ``n_classes`` channels. The regression branch ends with
    ``4 * reg_max`` channels. These hold one distribution over
    ``reg_max`` distance bins for each side of the box.
    `PrecisionBBoxHead` decodes them with `DFL` when ``reg_max`` is
    above ``1``.

    Example:
        >>> import torch
        >>> block = PreciseDecoupledBlock(8, 4, 4, n_classes=3, reg_max=4)
        >>> [tuple(t.shape) for t in block(torch.zeros(1, 8, 5, 5))]
        [(1, 19, 5, 5), (1, 3, 5, 5), (1, 16, 5, 5)]

    """

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
        """Initialize the classification and regression branches.

        Args:
            in_channels (int): Number of channels of the input feature
                map.
            reg_channels (int): Number of hidden channels of the
                regression branch.
            cls_channels (int): Number of hidden channels of the
                classification branch.
            n_classes (int): Number of classes, which is the number of
                output channels of the classification branch.
            reg_max (int): Number of distance bins for each side of a
                box. The regression branch outputs ``4 * reg_max``
                channels.

        """
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
        """Run both branches on one feature map.

        Args:
            x (``Tensor``): Feature map of shape
                ``[B, in_channels, H, W]``.

        Returns:
            ``tuple[Tensor, Tensor, Tensor]``: The tuple ``(features,
            classes, regressions)``. ``regressions`` holds the distance
            bin logits, of shape ``[B, 4 * reg_max, H, W]``. ``classes``
            holds the class logits, of shape ``[B, n_classes, H, W]``.
            ``features`` is ``regressions`` and ``classes`` concatenated
            along the channel axis, of shape
            ``[B, 4 * reg_max + n_classes, H, W]``.

        """
        regressions = self.regression_branch(x)
        classes = self.classification_branch(x)
        features = torch.cat([regressions, classes], dim=1)
        return features, classes, regressions


class EfficientDecoupledBlock(nn.Module):
    """Decoupled head of `EfficientBBoxHead` for one scale.

    A ``1x1`` `ConvBlock` with batch norm and SiLU first decodes the
    input. Two parallel branches then read the decoded map. Each branch
    is a ``3x3`` `ConvBlock` with batch norm and SiLU, followed by a
    ``1x1`` convolution. The classification branch ends with
    ``n_classes`` channels, the regression branch with ``4`` channels.

    Example:
        >>> import torch
        >>> block = EfficientDecoupledBlock(8, n_classes=3)
        >>> [tuple(t.shape) for t in block(torch.zeros(1, 8, 5, 5))]
        [(1, 8, 5, 5), (1, 3, 5, 5), (1, 4, 5, 5)]

    """

    __call__: Callable[[Tensor], tuple[Tensor, Tensor, Tensor]]

    @typechecked
    def __init__(self, in_channels: int, n_classes: int):
        """Initialize the decoder and the two branches.

        Args:
            in_channels (int): Number of channels of the input feature
                map. The decoder and the hidden layer of each branch
                keep this width.
            n_classes (int): Number of classes, which is the number of
                output channels of the classification branch.

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
        """Decode one feature map and run both branches on it.

        Args:
            x (``Tensor``): Feature map of shape
                ``[B, in_channels, H, W]``.

        Returns:
            ``tuple[Tensor, Tensor, Tensor]``: The tuple ``(features,
            classes, regressions)``. ``features`` is the decoded map,
            of shape ``[B, in_channels, H, W]``. ``classes`` holds the
            class logits, of shape ``[B, n_classes, H, W]``.
            ``regressions`` holds the box regression, of shape
            ``[B, 4, H, W]``.

        """
        features = self.decoder(x)

        classes = self.class_branch(features)
        regressions = self.regression_branch(features)

        return features, classes, regressions


class SegProto(nn.Sequential):
    """Mask prototype generator of `PrecisionSegmentBBoxHead`.

    The stack is a ``3x3`` `ConvBlock`, a ``2x2`` transposed convolution
    with stride ``2``, a second ``3x3`` `ConvBlock`, and a ``1x1``
    `ConvBlock`. Every `ConvBlock` uses batch norm and SiLU. An input of
    shape ``[B, in_channels, H, W]`` becomes a tensor of shape
    ``[B, out_channels, 2 * H, 2 * W]``.

    Example:
        >>> import torch
        >>> proto = SegProto(4, mid_channels=8, out_channels=2)
        >>> proto(torch.zeros(1, 4, 8, 8)).shape
        torch.Size([1, 2, 16, 16])

    """

    @typechecked
    def __init__(
        self, in_channels: int, mid_channels: int = 256, out_channels: int = 32
    ):
        """Initialize the prototype stack.

        Args:
            in_channels (int): Number of input channels.
            mid_channels (int): Number of channels of the three hidden
                layers. Defaults to ``256``.
            out_channels (int): Number of prototype masks, which is the
                number of output channels. Defaults to ``32``.

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
    r"""Decoder of the distribution focal loss (DFL) box regression.

    A DFL head predicts, for each of the four sides of a box, a
    distribution over ``reg_max`` integer distance bins. This module
    turns the bin logits :math:`z` of one side into the expected
    distance of that side. A softmax over the bins gives the
    probabilities. A frozen ``1x1`` convolution with the weights
    ``0, 1, ..., reg_max - 1`` then takes their weighted sum:

    .. math::

        d = \sum_{i=0}^{R - 1} i \cdot \text{softmax}(z)_i

    where :math:`R` is ``reg_max``. The result is a real value between
    ``0`` and ``reg_max - 1``, in the unit of one bin.

    Example:
        >>> import torch
        >>> dfl = DFL(reg_max=4)
        >>> x = torch.zeros(1, 16, 1, 1)
        >>> x[0, [2, 5, 8, 15]] = 1000.0  # one confident bin per side
        >>> dfl(x).flatten().tolist()
        [2.0, 1.0, 0.0, 3.0]

        A uniform distribution decodes to the mean bin index:

        >>> dfl(torch.zeros(1, 16, 2, 2)).unique().tolist()
        [1.5]

    """

    @typechecked
    def __init__(self, reg_max: int = 16):
        """Initialize the decoder and freeze its projection weights.

        Args:
            reg_max (int): Number of distance bins for each side of a
                box. Defaults to ``16``.

        """
        super().__init__()
        self.conv = nn.Conv2d(reg_max, 1, kernel_size=1, bias=False)
        self.conv.weight.data.copy_(
            torch.arange(reg_max, dtype=torch.float32).view(1, reg_max, 1, 1)
        )
        self.conv.requires_grad_(False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: Tensor) -> Tensor:
        """Decode the bin logits of every side into a distance.

        Args:
            x (``Tensor``): Logits of shape ``[B, 4 * reg_max, H, W]``.
                The channels hold the ``reg_max`` bins of the first
                side, then the bins of the second side, and so on.

        Returns:
            ``Tensor``: Expected distance of each side, of shape
            ``[B, 4, H, W]``.

        """
        n, _, h, w = x.size()
        x = x.view(n, 4, -1, h * w).permute(0, 2, 1, 3)
        x = self.softmax(x)
        return self.conv(x)[:, 0].view(n, 4, h, w)


class ConvBlock(nn.Module):
    """A 2D convolution, an optional batch norm, and an activation.

    The constructor stores every convolution argument under its own
    name, so `GeneralReparameterizableBlock` can read the geometry back
    when it fuses the block.

    Attributes:
        conv (``nn.Conv2d``): The convolution.
        bn (``nn.BatchNorm2d | None``): The batch norm, or ``None`` when
            ``use_norm`` is ``False``.
        activation (``Callable[[Tensor], Tensor]``): The activation that
            runs last. `torch.nn.Identity` when the constructor got
            ``False`` or ``None``.

    Example:
        >>> import torch
        >>> block = ConvBlock(3, 8, kernel_size=3, padding=1)
        >>> block(torch.zeros(1, 3, 16, 16)).shape
        torch.Size([1, 8, 16, 16])
        >>> ConvBlock(3, 8, kernel_size=1, use_norm=False).bn is None
        True

    """

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
        activation: Callable[[Tensor], Tensor] | bool | None = True,
        use_norm: bool = True,
        norm_momentum: float = 0.1,
    ):
        """Build the convolution, the batch norm, and the activation.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int | tuple[int, int]): Size of the kernel.
            stride (int | tuple[int, int]): Stride of the convolution.
                Defaults to ``1``.
            padding (int | tuple[int, int] | str): Padding of the
                convolution, or the string ``"same"`` or ``"valid"``.
                Defaults to ``0``.
            dilation (int | tuple[int, int]): Dilation of the kernel.
                Defaults to ``1``.
            groups (int): Number of groups of the convolution. Defaults
                to ``1``.
            bias (bool): Whether the convolution has a bias term.
                Defaults to ``False``.
            activation (``Callable[[Tensor], Tensor] | bool | None``):
                The activation. ``True`` selects `torch.nn.ReLU`.
                ``False`` or ``None`` selects `torch.nn.Identity`. Any
                other callable runs unchanged. Defaults to ``True``.
            use_norm (bool): Whether to add a batch norm after the
                convolution. Defaults to ``True``.
            norm_momentum (float): Momentum of the batch norm. Defaults
                to ``0.1``.

        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self._bias = bias

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
        """Apply the convolution, the batch norm, and the activation.

        Args:
            x (``Tensor``): Input of shape ``[B, in_channels, H, W]``.

        Returns:
            ``Tensor``: Output of shape ``[B, out_channels, H', W']``.
            The kernel size, stride, padding, and dilation set ``H'``
            and ``W'`` as in `torch.nn.Conv2d`.

        """
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        return self.activation(x)


class SqueezeExciteBlock(nn.Sequential):
    """Squeeze-and-excite channel attention.

    The block averages every channel over the spatial axes. Two ``1x1``
    convolutions with an activation between them map the averages to
    one value per channel. A sigmoid, or a hard sigmoid when
    ``hard_sigmoid`` is ``True``, maps these values into ``[0, 1]``.
    The block then multiplies the input by these per-channel gates.

    References:
        - Paper: `Squeeze-and-Excitation Networks <https://arxiv.org/pdf/1709.01507.pdf>`_
        - Code adapted from `ml-mobileone <https://github.com/apple/ml-mobileone/blob/main/mobileone.py>`_

    Example:
        >>> import torch
        >>> block = SqueezeExciteBlock(8, intermediate_channels=2)
        >>> block(torch.ones(1, 8, 4, 4)).shape
        torch.Size([1, 8, 4, 4])

    """

    @typechecked
    def __init__(
        self,
        in_channels: int,
        intermediate_channels: int,
        hard_sigmoid: bool = False,
        activation: nn.Module | None = None,
    ):
        """Initialize the squeeze-and-excite layers.

        Args:
            in_channels (int): Number of input channels, which the block
                also outputs.
            intermediate_channels (int): Number of channels between the
                two ``1x1`` convolutions.
            hard_sigmoid (bool): Whether to gate with
                `torch.nn.Hardsigmoid` instead of `torch.nn.Sigmoid`.
                Defaults to ``False``.
            activation (``nn.Module | None``): Activation between the
                two convolutions. ``None`` selects `torch.nn.ReLU`.
                Defaults to ``None``.

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
        """Scale every channel of ``x`` by its excitation gate.

        Args:
            x (``Tensor``): Input of shape ``[B, in_channels, H, W]``.

        Returns:
            ``Tensor``: ``x`` multiplied by gates of shape
            ``[B, in_channels, 1, 1]``, so of the same shape as ``x``.

        """
        return x * super().forward(x)


# TODO: Maybe a better name?
class GeneralReparameterizableBlock(Reparameterizable):
    """RepVGG-style block with parallel training branches.

    Before `reparameterize`, the block sums the outputs of up to three
    kinds of branches:

    - ``n_branches`` dense convolutions of size ``kernel_size``, each
      with a batch norm;
    - one ``1x1`` scale convolution with a batch norm, when
      ``use_scale_layer`` is ``True``;
    - one identity branch that is only a batch norm, when the input and
      output channels match and the stride is ``1``.

    The optional ``refine_block`` and the activation follow the sum.
    `reparameterize` folds every branch into ``fused_branch``, a single
    convolution with the same output. `restore` drops the fused
    convolution and returns to the branches. `BaseNode` calls
    `reparameterize` when it enters export mode and `restore` when it
    leaves it.

    Attributes:
        branches (list[ConvBlock]): The dense branches, stored in a
            `torch.nn.ModuleList`.
        scale_layer (ConvBlock | None): The ``1x1`` scale branch, or
            ``None`` when ``use_scale_layer`` is ``False``.
        skip_layer (``nn.BatchNorm2d | None``): The identity branch, or
            ``None`` when the block has no identity branch.
        refine_block (``nn.Module``): The block that runs on the sum of
            the branches. `torch.nn.Identity` when the constructor got
            ``None``.
        activation (``nn.Module``): The final activation.
        fused_branch (``nn.Conv2d | None``): The fused convolution, or
            ``None`` before `reparameterize` and after `restore`.

    See Also:
        `RepVGG reference implementation <https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py>`_.

    Example:
        >>> import torch
        >>> block = GeneralReparameterizableBlock(4, 4, n_branches=2).eval()
        >>> x = torch.linspace(-1, 1, 256).view(1, 4, 8, 8)
        >>> before = block(x)
        >>> block.reparameterize()
        >>> block.fused_branch is not None
        True
        >>> torch.allclose(before, block(x), atol=1e-6)
        True
        >>> block.restore()
        >>> block.fused_branch is None
        True

    """

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
        activation: nn.Module | bool | None = True,
    ):
        """Initialize the branches, the refinement, and the activation.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Size of the dense kernels. Defaults to
                ``3``.
            stride (int): Stride of the dense and scale branches. The
                identity branch exists only when the stride is ``1``.
                Defaults to ``1``.
            padding (int): Padding of the dense branches. Defaults to
                ``1``.
            groups (int): Number of groups of every convolution.
                Defaults to ``1``.
            n_branches (int): Number of dense branches. `reparameterize`
                fuses all of them into one convolution. Defaults to
                ``1``.
            refine_block (``nn.Module | Literal["se"] | None``): Block
                that runs on the sum of the branches, before the
                activation. A module runs unchanged. The string
                ``"se"`` builds a `SqueezeExciteBlock` with
                ``out_channels // 16`` intermediate channels. ``None``
                applies no refinement. Defaults to ``None``.
            use_scale_layer (bool): Whether to add the ``1x1`` scale
                branch. Defaults to ``True``.
            scale_layer_padding (int | tuple[int, int] | None): Padding
                of the scale branch. ``None`` or ``0`` selects
                ``padding - kernel_size // 2``. For an odd
                ``kernel_size``, this gives the scale branch the same
                output size as the dense branches. Defaults to
                ``None``.
            activation (``nn.Module | bool | None``): The final
                activation. ``True`` selects `torch.nn.ReLU`. ``False``
                or ``None`` selects `torch.nn.Identity`. Any other
                module runs unchanged. Defaults to ``True``.

        """
        super().__init__()

        self._in_channels = in_channels
        self._out_channels = out_channels
        self._kernel_size = kernel_size
        self._groups = groups

        self.skip_layer: nn.BatchNorm2d | None = None
        if out_channels == in_channels and stride in (1, (1, 1)):
            self.skip_layer = nn.BatchNorm2d(in_channels)

        self.scale_layer: ConvBlock | None = None

        if use_scale_layer:
            padding_scale = scale_layer_padding or padding - kernel_size // 2
            self.scale_layer = ConvBlock(
                in_channels=self._in_channels,
                out_channels=self._out_channels,
                kernel_size=1,
                stride=stride,
                padding=padding_scale,
                groups=self._groups,
                activation=False,
            )

        branches = [
            ConvBlock(
                in_channels=self._in_channels,
                out_channels=self._out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=self._groups,
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
        """Run the block in its current state.

        Args:
            x (``Tensor``): Input of shape ``[B, in_channels, H, W]``.

        Returns:
            ``Tensor``: Output of shape ``[B, out_channels, H', W']``.
            The kernel size, stride, and padding of the dense branches
            set ``H'`` and ``W'`` as in `torch.nn.Conv2d`. Before
            `reparameterize`, the output is the activation of the
            refined sum of all branches. After it, the fused convolution
            replaces the sum.

        """
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
        """The name of the class of this block."""
        return self.__class__.__name__

    @override
    def reparameterize(self) -> None:
        """Fuse every branch into a single convolution.

        The method turns each branch into one kernel and one bias:

        - a dense branch gives its kernel with the batch norm folded
          in;
        - the scale branch gives its ``1x1`` kernel, padded by
          ``kernel_size // 2`` on each side, with the batch norm folded
          in;
        - the identity branch gives a centered unit kernel with the
          batch norm folded in.

        The scale fold needs an odd ``kernel_size``. ``fused_branch`` is
        a `torch.nn.Conv2d` with the geometry of the first dense branch.
        Its weight is the sum of these kernels. Its bias is the sum of
        these biases. `forward` then uses ``fused_branch`` instead of
        the branches. The training branches stay in the module
        unchanged.

        The fold reads the running statistics of every batch norm. The
        fused output therefore equals the branch sum in the eval state
        only. A second call does nothing while ``fused_branch`` exists.

        """
        if self.fused_branch is not None:
            return

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
        """Drop the fused convolution and return to the branches.

        The method detaches the parameters of ``fused_branch``, removes
        the submodule, and sets ``fused_branch`` to ``None``. `forward`
        then sums the training branches again. They hold the same
        parameters as before `reparameterize`. A call without a fused
        branch does nothing.

        """
        if self.fused_branch is None:
            return

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
            pad = self._kernel_size // 2
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
        input_dim = self._in_channels // self._groups
        kernel = torch.zeros(
            (
                self._in_channels,
                input_dim,
                self._kernel_size,
                self._kernel_size,
            ),
            dtype=module.weight.dtype,
            device=module.weight.device,
        )
        for i in range(self._in_channels):
            kernel[
                i,
                i % input_dim,
                self._kernel_size // 2,
                self._kernel_size // 2,
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
                "provided for reparameterization."
            )
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1).to(kernel.device)
        return kernel * t, beta - running_mean * gamma / std


class BlockRepeater(nn.Sequential):
    """Sequential stack of ``n_repeats`` blocks from one factory.

    The first block receives the keyword arguments as given. When the
    arguments contain ``out_channels``, every block after the first
    receives ``in_channels`` equal to that ``out_channels``. A stack can
    therefore change the number of channels in its first block and keep
    it in the rest.

    Example:
        >>> import torch
        >>> kwargs = {"in_channels": 3, "out_channels": 8, "kernel_size": 1}
        >>> stack = BlockRepeater(ConvBlock, n_repeats=3, **kwargs)
        >>> [block.in_channels for block in stack]
        [3, 8, 8]
        >>> stack(torch.zeros(1, 3, 4, 4)).shape
        torch.Size([1, 8, 4, 4])

    """

    @typechecked
    def __init__(
        self, module: Callable[..., nn.Module], /, *, n_repeats: int, **kwargs
    ):
        """Build the blocks and register them in order.

        Args:
            module (``Callable[..., nn.Module]``): Factory that returns
                one block. The stack calls it once for each block.
            n_repeats (int): Number of blocks. A value below ``1`` still
                builds one block.
            **kwargs (``Any``): Keyword arguments forwarded to
                ``module``. When they contain ``out_channels``, the
                blocks after the first receive ``in_channels`` equal to
                ``out_channels``.

        """
        blocks = [module(**kwargs)]

        if "out_channels" in kwargs:
            kwargs["in_channels"] = kwargs["out_channels"]

        blocks.extend(module(**kwargs) for _ in range(n_repeats - 1))

        super().__init__(*blocks)


class CSPStackRepBlock(nn.Module):
    """CSP block with a stack of `BottleRep` blocks on one of its paths.

    Two ``1x1`` `ConvBlock` layers each map the input to a path of
    ``int(out_channels * e)`` channels. The first path runs through a
    `BlockRepeater` of `BottleRep` blocks. The block concatenates both
    paths along the channel axis and maps them to ``out_channels`` with
    a third ``1x1`` `ConvBlock`. The three ``1x1`` `ConvBlock` layers
    use batch norm and ReLU.

    Example:
        >>> import torch
        >>> block = CSPStackRepBlock(8, 16, n_blocks=2)
        >>> len(block.rep_stack)
        1
        >>> block(torch.zeros(1, 8, 4, 4)).shape
        torch.Size([1, 16, 4, 4])

    """

    @typechecked
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_blocks: int = 1,
        e: float = 0.5,
    ):
        """Initialize the three convolutions and the stack.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            n_blocks (int): Controls the number of RepVGG blocks in the
                stack. Every `BottleRep` holds two of them, so the stack
                has ``max(1, n_blocks // 2)`` bottlenecks and
                ``2 * max(1, n_blocks // 2)`` RepVGG blocks. Only an even
                value of at least ``2`` gives ``n_blocks`` RepVGG blocks.
                Defaults to ``1``.
            e (float): Fraction of ``out_channels`` that each path
                carries. Defaults to ``0.5``.

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
        """Run both paths and merge them.

        Args:
            x (``Tensor``): Input of shape ``[B, in_channels, H, W]``.

        Returns:
            ``Tensor``: Output of shape ``[B, out_channels, H, W]``.

        """
        out_1 = self.conv_1(x)
        out_1 = self.rep_stack(out_1)
        out_2 = self.conv_2(x)
        out = torch.cat([out_1, out_2], dim=1)
        return self.conv_3(out)


class BottleRep(nn.Module):
    """Two blocks from one factory with a weighted residual connection.

    The default factory is `GeneralReparameterizableBlock`, which makes
    the pair two RepVGG-style blocks. The residual connection exists
    only when ``in_channels`` equals ``out_channels``. Its weight
    ``alpha`` is a learnable parameter of shape ``[1]`` or the constant
    ``1.0``.

    Example:
        >>> import torch
        >>> BottleRep(8, 16, weight=False).alpha
        1.0
        >>> BottleRep(8, 16)(torch.zeros(1, 8, 4, 4)).shape
        torch.Size([1, 16, 4, 4])

    """

    @typechecked
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        module: ModuleFactory = GeneralReparameterizableBlock,
        weight: bool = True,
        **kwargs,
    ):
        """Initialize the two blocks and the shortcut weight.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels of both
                blocks.
            module (ModuleFactory): Factory of the two blocks. The
                first block maps ``in_channels`` to ``out_channels``,
                the second keeps ``out_channels``. Defaults to
                `GeneralReparameterizableBlock`.
            weight (bool): Whether ``alpha`` is a learnable
                ``nn.Parameter`` of shape ``[1]`` with the initial value
                ``1.0``. Otherwise ``alpha`` is the constant ``1.0``.
                Defaults to ``True``.
            **kwargs (``Any``): Keyword arguments forwarded to both
                calls of ``module``.

        """
        super().__init__()
        self.conv_1 = module(
            in_channels=in_channels, out_channels=out_channels, **kwargs
        )
        self.conv_2 = module(
            in_channels=out_channels, out_channels=out_channels, **kwargs
        )
        self._shortcut = in_channels == out_channels
        self.alpha = nn.Parameter(torch.ones(1)) if weight else 1.0

    def forward(self, x: Tensor) -> Tensor:
        """Apply both blocks and add the weighted shortcut.

        Args:
            x (``Tensor``): Input of shape ``[B, in_channels, H, W]``.

        Returns:
            ``Tensor``: ``conv_2(conv_1(x)) + alpha * x`` when the
            channel counts match, else ``conv_2(conv_1(x))``. With the
            default blocks the shape is ``[B, out_channels, H, W]``.

        """
        out = self.conv_1(x)
        out = self.conv_2(out)
        return out + self.alpha * x if self._shortcut else out


class SpatialPyramidPoolingBlock(nn.Module):
    """Fast spatial pyramid pooling (SPPF) with one shared max-pool.

    A ``1x1`` `ConvBlock` halves the channels. The block then applies
    the same max-pool three times in a row. With :math:`k` equal to
    ``kernel_size``, this equals pooling with windows of size
    :math:`k`, :math:`2k - 1`, and :math:`3k - 2`. The block
    concatenates the halved map and the three pooled maps, and a second
    ``1x1`` `ConvBlock` maps them to ``out_channels``. Both `ConvBlock`
    layers use batch norm and ReLU.

    Example:
        >>> import torch
        >>> spp = SpatialPyramidPoolingBlock(8, 16)
        >>> spp(torch.zeros(1, 8, 8, 8)).shape
        torch.Size([1, 16, 8, 8])

    """

    @typechecked
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int = 5
    ):
        """Initialize the two convolutions and the max-pool.

        Args:
            in_channels (int): Number of input channels. The hidden
                width is ``in_channels // 2``.
            out_channels (int): Number of output channels.
            kernel_size (int): Size of the max-pool window. The pool
                uses stride ``1`` and padding ``kernel_size // 2``, so an
                odd size keeps the spatial size. Defaults to ``5``.

        """
        super().__init__()

        intermediate_channels = in_channels // 2  # hidden channels
        self.conv1 = ConvBlock(in_channels, intermediate_channels, 1, 1)
        self.conv2 = ConvBlock(intermediate_channels * 4, out_channels, 1, 1)
        self.max_pool = nn.MaxPool2d(
            kernel_size=kernel_size, stride=1, padding=kernel_size // 2
        )

    def forward(self, x: Tensor) -> Tensor:
        """Pool at three scales and merge the results.

        Args:
            x (``Tensor``): Input of shape ``[B, in_channels, H, W]``.

        Returns:
            ``Tensor``: Output of shape ``[B, out_channels, H, W]`` for
            an odd ``kernel_size``.

        """
        x = self.conv1(x)
        # apply max-pooling at three different scales
        y1 = self.max_pool(x)
        y2 = self.max_pool(y1)
        y3 = self.max_pool(y2)

        x = torch.cat([x, y1, y2, y3], dim=1)
        return self.conv2(x)


class AttentionRefinementBlock(nn.Module):
    """Attention refinement module (ARM) of BiSeNet V1.

    A ``3x3`` `ConvBlock` with batch norm and ReLU maps the input to
    ``out_channels``. A global average pool and a ``1x1`` `ConvBlock`
    with batch norm and sigmoid turn the result into one gate per
    channel. The block multiplies the convolved map by its gates.

    References:
        - Code adapted from `BiseNetv1 <https://github.com/taveraantonio/BiseNetv1>`_

    Example:
        >>> import torch
        >>> block = AttentionRefinementBlock(8, 16)
        >>> block(torch.zeros(2, 8, 4, 4)).shape
        torch.Size([2, 16, 4, 4])

    """

    @typechecked
    def __init__(self, in_channels: int, out_channels: int):
        """Initialize the convolution and the attention branch.

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
        """Convolve ``x`` and scale the result by its channel gates.

        Args:
            x (``Tensor``): Input of shape ``[B, in_channels, H, W]``.

        Returns:
            ``Tensor``: Output of shape ``[B, out_channels, H, W]``.

        """
        x = self.conv(x)
        attention = self.attention(x)
        return x * attention


class FeatureFusionBlock(nn.Module):
    """Feature fusion module (FFM) of BiSeNet V1.

    The block concatenates two feature maps along the channel axis and
    maps them to ``out_channels`` with a ``1x1`` `ConvBlock` with batch
    norm and ReLU. An attention branch turns the fused map into one
    gate per channel. The branch runs, in order:

    - a global average pool;
    - a ``1x1`` `ConvBlock` with batch norm and ReLU;
    - a ``1x1`` `ConvBlock` with batch norm and no activation;
    - a sigmoid.

    The output is the fused map plus the fused map multiplied by its
    gates.

    References:
        - Code adapted from `BiseNetv1 <https://github.com/taveraantonio/BiseNetv1>`_

    Example:
        >>> import torch
        >>> block = FeatureFusionBlock(8, 16)
        >>> x1, x2 = torch.zeros(2, 3, 4, 4), torch.zeros(2, 5, 4, 4)
        >>> block(x1, x2).shape
        torch.Size([2, 16, 4, 4])

    """

    @typechecked
    def __init__(
        self, in_channels: int, out_channels: int, reduction: int = 1
    ):
        """Initialize the fusion convolution and the attention branch.

        Args:
            in_channels (int): Number of channels of both inputs
                together.
            out_channels (int): Number of output channels.
            reduction (int): Divisor of ``out_channels`` inside the
                attention branch. Only ``1`` works. Any other value
                makes `forward` fail, because the second attention
                convolution expects ``out_channels`` input channels.
                Defaults to ``1``.

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
        """Fuse two feature maps of the same spatial size.

        Args:
            x1 (``Tensor``): First map of shape ``[B, C1, H, W]``.
            x2 (``Tensor``): Second map of shape ``[B, C2, H, W]``, with
                ``C1 + C2`` equal to ``in_channels``.

        Returns:
            ``Tensor``: Fused map of shape ``[B, out_channels, H, W]``.

        """
        fusion = torch.cat([x1, x2], dim=1)
        x = self.conv_1x1(fusion)
        attention = self.attention(x)
        return x + x * attention


class UpscaleOnline(nn.Module):
    """Interpolation of a tensor to a size given in the forward pass.

    The constructor stores only the interpolation mode. The target
    height and width are arguments of `forward`.

    Example:
        >>> import torch
        >>> up = UpscaleOnline("nearest")
        >>> up(torch.tensor([[[[1.0, 2.0]]]]), 1, 4).tolist()
        [[[[1.0, 1.0, 2.0, 2.0]]]]
        >>> UpscaleOnline()(torch.zeros(1, 2, 4, 4), 8, 6).shape
        torch.Size([1, 2, 8, 6])

    """

    @typechecked
    def __init__(self, mode: str = "bilinear"):
        """Store the interpolation mode.

        Args:
            mode (str): Interpolation mode of
                `torch.nn.functional.interpolate`, for example
                ``"nearest"`` or ``"bilinear"``. Defaults to
                ``"bilinear"``.

        """
        super().__init__()
        self._mode = mode

    def forward(
        self, x: Tensor, output_height: int, output_width: int
    ) -> Tensor:
        """Resize ``x`` to ``output_height`` by ``output_width``.

        Args:
            x (``Tensor``): Input of shape ``[B, C, H, W]``.
            output_height (int): Height of the output.
            output_width (int): Width of the output.

        Returns:
            ``Tensor``: Output of shape
            ``[B, C, output_height, output_width]``.

        """
        return F.interpolate(
            x, size=[output_height, output_width], mode=self._mode
        )


class DropPath(nn.Module):
    """Stochastic depth that drops the residual path of whole samples.

    In the training state, the module zeroes each sample of the batch
    with probability ``drop_prob``. When ``scale_by_keep`` is ``True``,
    it divides the kept samples by ``1 - drop_prob``. The expected
    value of the output then stays equal to the input. In the eval
    state, or when ``drop_prob`` is ``0.0``, the module returns its
    input unchanged.

    Place the module on the residual path of a block, as in
    ``x + drop_path(block(x))``.

    Notes:
        License: `Apache License 2.0 <https://github.com/huggingface/pytorch-image-models?tab=Apache-2.0-1-ov-file#readme>`_.

    See Also:
        `Original code (TIMM) <https://github.com/rwightman/pytorch-image-models>`_.

    Example:
        >>> import torch
        >>> x = torch.ones(2, 3)
        >>> torch.equal(DropPath(drop_prob=0.5).eval()(x), x)
        True
        >>> DropPath(drop_prob=1.0).drop_path(x).tolist()
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        >>> out = DropPath(drop_prob=0.5).drop_path(x)
        >>> set(out.unique().tolist()) <= {0.0, 2.0}
        True

    """

    @typechecked
    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        """Store the drop probability and the scaling flag.

        Args:
            drop_prob (float): Probability that the module zeroes a
                sample, in ``[0, 1]``. Defaults to ``0.0``.
            scale_by_keep (bool): Whether to divide the kept samples by
                ``1 - drop_prob``. Defaults to ``True``.

        """
        super().__init__()
        self._drop_prob = drop_prob
        self._scale_by_keep = scale_by_keep

    def drop_path(self, x: Tensor) -> Tensor:
        """Drop samples regardless of the training state.

        Args:
            x (``Tensor``): Input of shape ``[B, ...]``.

        Returns:
            ``Tensor``: ``x`` multiplied by a mask of shape
            ``[B, 1, ..., 1]``. Each mask entry is ``0`` with
            probability ``drop_prob``. Otherwise it is
            ``1 / (1 - drop_prob)`` when ``scale_by_keep`` is ``True``
            and ``drop_prob`` is below ``1.0``, else ``1``.

        """
        keep_prob = 1 - self._drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self._scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor

    def forward(self, x: Tensor) -> Tensor:
        """Apply `drop_path` in the training state only.

        Args:
            x (``Tensor``): Input of shape ``[B, ...]``.

        Returns:
            ``Tensor``: ``x`` unchanged when ``drop_prob`` is ``0.0`` or
            the module is in the eval state. Otherwise the result of
            `drop_path`.

        """
        if self._drop_prob == 0.0 or not self.training:
            return x
        return self.drop_path(x)


class ConvStack(BlockRepeater):
    """Stack of ``3x3`` `ConvBlock` layers with batch norm and ReLU.

    The first block maps ``in_channels`` to ``out_channels``. The others
    keep ``out_channels``. Padding ``1`` keeps the spatial size.

    Example:
        >>> import torch
        >>> stack = ConvStack(3, 8, n_repeats=3)
        >>> [block.in_channels for block in stack]
        [3, 8, 8]
        >>> stack(torch.zeros(1, 3, 16, 16)).shape
        torch.Size([1, 8, 16, 16])

    """

    def __init__(
        self, in_channels: int, out_channels: int, *, n_repeats: int = 2
    ):
        """Build the stack through `BlockRepeater`.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            n_repeats (int): Number of `ConvBlock` layers. Defaults to
                ``2``.

        """
        super().__init__(
            ConvBlock,
            n_repeats=n_repeats,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
        )
