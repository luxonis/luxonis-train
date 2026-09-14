"""The blocks of the PPLCNetV3 backbone."""

from contextlib import suppress

import torch
from torch import Tensor, nn
from typeguard import typechecked

from luxonis_train.nodes.blocks import (
    GeneralReparameterizableBlock,
    SqueezeExciteBlock,
)


class AffineActivation(nn.Module):
    """``Hardswish`` followed by a learnable affine map.

    The output is ``scale * hardswish(x) + bias``. An `AffineBlock`
    holds ``scale`` and ``bias``. They start at ``1.0`` and ``0.0``, so
    the module starts as a plain `torch.nn.Hardswish`.

    Example:
        >>> import torch
        >>> act = AffineActivation()
        >>> act(torch.tensor([-4.0, 0.0, 4.0])).tolist()
        [0.0, 0.0, 4.0]

    """

    def __init__(self):
        """Initialize the ``Hardswish`` and the `AffineBlock`."""
        super().__init__()
        self.activation = nn.Hardswish()
        self.affine = AffineBlock()

    def forward(self, x: Tensor) -> Tensor:
        """Apply ``Hardswish``, then the affine map.

        Args:
            x (``Tensor``): A tensor of any shape.

        Returns:
            ``Tensor``: ``scale * hardswish(x) + bias``, of the same
            shape as ``x``.

        """
        # WARN: Is the order correct (activation -> affine)?
        return self.affine(self.activation(x))


class AffineBlock(nn.Module):
    """Learnable affine map ``scale * x + bias`` with scalar parameters.

    ``scale`` and ``bias`` are ``torch.nn.Parameter`` objects of shape
    ``[1]``. The same two values apply to all elements of the input.

    Example:
        >>> import torch
        >>> block = AffineBlock(scale_value=2.0, bias_value=1.0)
        >>> block(torch.tensor([1.0, 2.0])).tolist()
        [3.0, 5.0]

    """

    @typechecked
    def __init__(self, scale_value: float = 1.0, bias_value: float = 0.0):
        """Initialize the scale and the bias parameters.

        Args:
            scale_value (float): The initial value of ``scale``. Give a
                ``float``. An ``int`` passes the type check, but it makes
                an integer tensor. Then the constructor raises
                ``RuntimeError``.
            bias_value (float): The initial value of ``bias``. The same
                rule applies.

        """
        super().__init__()

        self.scale = nn.Parameter(torch.full((1,), scale_value))
        self.bias = nn.Parameter(torch.full((1,), bias_value))

    def forward(self, x: Tensor) -> Tensor:
        """Apply the affine map.

        Args:
            x (``Tensor``): A tensor of any shape.

        Returns:
            ``Tensor``: ``scale * x + bias``, of the same shape as ``x``.

        """
        return self.scale * x + self.bias


with suppress(ImportError):
    from aimet_torch.v2.nn import QuantizationMixin

    @QuantizationMixin.implements(AffineBlock)
    class QuantizedAffineBlock(QuantizationMixin, AffineBlock):
        """Quantized form of `AffineBlock` for AIMET.

        The class exists only when ``aimet_torch`` is installed. The
        decorator ``QuantizationMixin.implements`` registers it for
        `AffineBlock`. The block has one input quantizer and one output
        quantizer.

        """

        def __quant_init__(self):
            super().__quant_init__()

            # Declare the number of input/output quantizers
            self.input_quantizers = nn.ModuleList([None])  # type: ignore
            self.output_quantizers = nn.ModuleList([None])  # type: ignore

        def forward(self, x: Tensor) -> Tensor:
            """Run `AffineBlock` with quantized inputs and parameters.

            The method quantizes ``x`` with the input quantizer. It runs
            `AffineBlock.forward` with the quantized parameters in
            place. Then it quantizes the result with the output
            quantizer. It skips a quantizer that is ``None``.

            Args:
                x (``Tensor``): A tensor of any shape.

            Returns:
                ``Tensor``: ``scale * x + bias``, of the same shape as
                ``x``.

            """
            # Quantize input tensors
            if self.input_quantizers[0]:
                x = self.input_quantizers[0](x)

            # Run forward with quantized inputs and parameters
            with self._patch_quantized_parameters():
                ret = super().forward(x)

            # Quantize output tensors
            if self.output_quantizers[0]:
                ret = self.output_quantizers[0](ret)

            return ret


class LCNetV3Block(nn.Module):
    r"""Depthwise separable block of PPLCNetV3.

    The block runs these layers in order:

    - A :math:`k \times k` depthwise `GeneralReparameterizableBlock`. It
      has ``n_branches`` dense branches, a :math:`1 \times 1` scale
      branch, and an identity branch when ``stride`` is ``1``.
    - An optional `SqueezeExciteBlock` with ``in_channels // 4`` hidden
      channels, a ``ReLU``, and a hard sigmoid gate.
    - A :math:`1 \times 1` pointwise `GeneralReparameterizableBlock`. It
      has ``n_branches`` dense branches, no scale branch, and an
      identity branch when ``in_channels`` is equal to
      ``out_channels``.

    Each of the two convolutions applies an `AffineBlock` to the sum of
    its branches, then an `AffineActivation`. With a ``stride`` of
    ``2``, the depthwise convolution has no activation.

    Example:
        >>> import torch
        >>> block = LCNetV3Block(8, 16, kernel_size=3, stride=2, use_se=True)
        >>> block(torch.zeros(1, 8, 32, 32)).shape
        torch.Size([1, 16, 16, 16])

    """

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
        """Initialize the two convolutions and the optional attention.

        Args:
            in_channels (int): The number of input channels. The
                depthwise convolution keeps this number.
            out_channels (int): The number of output channels of the
                pointwise convolution.
            kernel_size (int): The kernel size of the depthwise
                convolution. The padding is ``(kernel_size - 1) // 2``.
                Use an odd value.
            stride (int): The stride of the depthwise convolution. ``2``
                also removes the activation after it.
            use_se (bool): Whether to add the `SqueezeExciteBlock`
                between the two convolutions.
            n_branches (int): The number of dense branches of each
                convolution.

        """
        super().__init__()
        self.dw_conv = GeneralReparameterizableBlock(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            groups=in_channels,
            n_branches=n_branches,
            refine_block=AffineBlock(),
            activation=AffineActivation() if stride != 2 else None,
        )
        if use_se:
            self.se = SqueezeExciteBlock(
                in_channels=in_channels,
                intermediate_channels=in_channels // 4,
                hard_sigmoid=True,
            )
        else:
            self.se = nn.Identity()

        self.pw_conv = GeneralReparameterizableBlock(
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
        """Apply the two convolutions and the optional attention.

        Args:
            x (``Tensor``): The input of shape ``[B, in_channels, H, W]``.

        Returns:
            ``Tensor``: The output of shape
            ``[B, out_channels, ceil(H / stride), ceil(W / stride)]``,
            for an odd ``kernel_size``.

        """
        return self.pw_conv(self.se(self.dw_conv(x)))


class LCNetV3Layer(nn.Sequential):
    """Sequence of `LCNetV3Block` blocks that forms one PPLCNetV3 layer.

    The layer builds one block for each position of the four lists.
    `scale_up` scales each value of ``out_channels`` by ``scale``. Each
    block takes the output channels of the block before it. The
    ``forward`` of `torch.nn.Sequential` runs the blocks in order.

    Attributes:
        in_channels (int): The number of input channels of the layer.
        out_channels (int): The number of output channels of the layer,
            ``scale_up(out_channels[-1], scale)``.

    Example:
        >>> import torch
        >>> layer = LCNetV3Layer(
        ...     16, [64, 64], [3, 3], [2, 1], [False, False], scale=0.95
        ... )
        >>> layer.out_channels, len(layer)
        (64, 2)
        >>> layer(torch.zeros(1, 16, 8, 8)).shape
        torch.Size([1, 64, 4, 4])

    """

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
        """Build one `LCNetV3Block` for each position of the lists.

        Args:
            in_channels (int): The number of input channels of the first
                block.
            out_channels (list[int]): The output channels of each block,
                before `scale_up` scales them.
            kernel_sizes (list[int]): The depthwise kernel size of each
                block.
            strides (list[int]): The depthwise stride of each block.
            use_se (list[bool]): Whether each block has a
                `SqueezeExciteBlock`.
            n_branches (int): The number of dense branches of each
                convolution of each block.
            scale (float): The width multiplier that `scale_up` applies
                to ``out_channels``.

        Raises:
            ValueError: When the four lists do not have the same length.

        """
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
    """Scale a channel count and round it to a multiple of ``divisor``.

    The function multiplies ``v`` by ``scale``. It rounds the product to
    the nearest multiple of ``divisor``, and a product halfway between
    two multiples rounds up. The result is at least ``min_value``. When
    the result is below 90% of the product, the function adds one
    ``divisor``.

    Args:
        v (float): The channel count to scale.
        scale (float): The multiplier.
        divisor (int): The multiple to round to.
        min_value (int | None): The smallest result. ``None`` selects
            ``divisor``.

    Returns:
        int: The scaled and rounded channel count.

    Example:
        >>> scale_up(512, 0.95)
        480

        ``10`` rounds to ``8``, which is below 90% of ``10``. Thus the
        function adds ``8``:

        >>> scale_up(10, 1.0, divisor=8)
        16

    """
    v = v * scale
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v
