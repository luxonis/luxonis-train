from contextlib import suppress

import torch
from torch import Tensor, nn
from typeguard import typechecked

from luxonis_train.nodes.blocks import (
    GeneralReparametrizableBlock,
    SqueezeExciteBlock,
)


class AffineActivation(nn.Module):
    """Affine Activation module."""

    def __init__(self):
        super().__init__()
        self.activation = nn.Hardswish()
        self.affine = AffineBlock()

    def forward(self, x: Tensor) -> Tensor:
        # WARN: Is the order correct (activation -> affine)?
        r"""Apply hard-swish followed by a learned affine transform.

        Args:
            x: Tensor whose elements will be activated and calibrated.

        Returns:
            Activated tensor with learned scaling and bias.

        .. shape-contract::

            Inputs
                :math:`x`
                    :math:`S`

            Outputs
                :math:`\mathrm{output}`
                    :math:`S`

            Symbols
                :math:`S`
                    Arbitrary tensor shape preserved by the operation.

        """
        return self.affine(self.activation(x))


class AffineBlock(nn.Module):
    """Affine Block module."""

    @typechecked
    def __init__(self, scale_value: float = 1.0, bias_value: float = 0.0):
        super().__init__()

        self.scale = nn.Parameter(torch.full((1,), scale_value))
        self.bias = nn.Parameter(torch.full((1,), bias_value))

    def forward(self, x: Tensor) -> Tensor:
        r"""Scale and shift every tensor element.

        Args:
            x: Tensor to scale and shift.

        Returns:
            Tensor with the learned scale and bias applied.

        .. shape-contract::

            Inputs
                :math:`x`
                    :math:`S`

            Outputs
                :math:`\mathrm{output}`
                    :math:`S`

            Symbols
                :math:`S`
                    Arbitrary tensor shape preserved by the operation.

        """
        return self.scale * x + self.bias


with suppress(ImportError):
    from aimet_torch.v2.nn import QuantizationMixin

    @QuantizationMixin.implements(AffineBlock)
    class QuantizedAffineBlock(QuantizationMixin, AffineBlock):
        """Quantized Affine Block module."""

        def __quant_init__(self):
            super().__quant_init__()

            # Declare the number of input/output quantizers
            self.input_quantizers = nn.ModuleList([None])  # type: ignore
            self.output_quantizers = nn.ModuleList([None])  # type: ignore

        def forward(self, x: Tensor) -> Tensor:
            # Quantize input tensors
            r"""Apply the quantized affine transform.

            Args:
                x: Tensor to quantize, scale, and shift.

            Returns:
                Quantized tensor with the affine transform applied.

            .. shape-contract::

                Inputs
                    :math:`x`
                        :math:`S`

                Outputs
                    :math:`\mathrm{output}`
                        :math:`S`

                Symbols
                    :math:`S`
                        Arbitrary tensor shape preserved by the operation.

            """
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
    """L C Net V3 Block module."""

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
        r"""Apply an LCNetV3 depthwise-pointwise block.

        Args:
            x: Feature map supplied to the LCNetV3 block.

        Returns:
            Feature map produced by the LCNetV3 block.

        .. shape-contract::

            Inputs
                :math:`x`
                    :math:`(B, C_{\mathrm{in}}, H, W)`

            Outputs
                :math:`\mathrm{output}`
                    :math:`(B, C_{\mathrm{out}}, H_{\mathrm{out}}, W_{\mathrm{out}})`

        """  # noqa: E501
        return self.pw_conv(self.se(self.dw_conv(x)))


class LCNetV3Layer(nn.Sequential):
    """L C Net V3 Layer module."""

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

    def forward(self, x: Tensor) -> Tensor:
        r"""Apply a sequence of LCNetV3 blocks.

        Args:
            x: Feature map supplied to the LCNetV3 block sequence.

        Returns:
            Output of the final LCNetV3 block.

        .. shape-contract::

            Inputs
                :math:`x`
                    :math:`(B, C_{\mathrm{in}}, H, W)`

            Outputs
                :math:`\mathrm{output}`
                    :math:`(B, C_{\mathrm{out}}, H_{\mathrm{out}}, W_{\mathrm{out}})`

        """  # noqa: E501
        return super().forward(x)


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
