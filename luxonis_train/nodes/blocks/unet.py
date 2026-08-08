from typing import Literal

import torch
from torch import Tensor, nn
from typeguard import typechecked

from .blocks import ConvBlock, ConvStack
from .utils import forward_gather


class EncoderBlock(nn.Sequential):
    """Optional max-pool followed by a stack of convolutions."""

    @typechecked
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_repeats: int,
        *,
        max_pool: bool = True,
    ):
        super().__init__(
            nn.MaxPool2d(2) if max_pool else nn.Identity(),
            ConvStack(in_channels, out_channels, n_repeats=n_repeats),
        )

    def forward(self, x: Tensor) -> Tensor:
        r"""Downsample features through one encoder stage.

        Args:
            x: Feature map to encode and downsample.

        Returns:
            Encoded feature map at the next lower resolution.

        .. shape-contract::

            Inputs
                :math:`x`
                    :math:`(B, C_{\mathrm{in}}, H, W)`

            Outputs
                :math:`\mathrm{output}`
                    :math:`(B, C_{\mathrm{out}}, H_{\mathrm{out}}, W_{\mathrm{out}})`

        """  # noqa: E501
        return super().forward(x)


class SimpleEncoder(nn.Sequential):
    """Encoder halving the resolution once per width multiplier."""

    @typechecked
    def __init__(
        self,
        in_channels: int,
        base_hidden_channels: int,
        width_multipliers: list[float],
        n_convolutions: int = 2,
    ):
        blocks = []
        for i, width_multiplier in enumerate(
            [*width_multipliers, width_multipliers[-1]]
        ):
            out_channels = int(base_hidden_channels * width_multiplier)
            blocks.append(
                EncoderBlock(
                    in_channels,
                    out_channels,
                    max_pool=i > 0,
                    n_repeats=n_convolutions,
                )
            )
            in_channels = out_channels
        super().__init__(*blocks)

    def forward(self, x: Tensor) -> Tensor:
        r"""Encode an image into a compact feature map.

        Args:
            x: Image batch or feature map to encode.

        Returns:
            Compact encoded feature map.

        .. shape-contract::

            Inputs
                :math:`x`
                    :math:`(B, C_{\mathrm{in}}, H, W)`

            Outputs
                :math:`\mathrm{output}`
                    :math:`(B, C_{\mathrm{out}}, H_{\mathrm{out}}, W_{\mathrm{out}})`

        """  # noqa: E501
        return super().forward(x)


class UNetEncoder(SimpleEncoder):
    """Encoder keeping every stage output for the decoder skips."""

    def forward(self, x: Tensor) -> list[Tensor]:
        r"""Collect features from each U-Net encoder stage.

        Args:
            x: Image batch or feature map entering the U-Net encoder.

        Returns:
            Feature maps collected before each downsampling stage.

        .. shape-contract::

            Inputs
                :math:`x`
                    :math:`(B, C_{\mathrm{in}}, H_{\mathrm{in}}, W_{\mathrm{in}})`

            Outputs
                :math:`\mathrm{features}_{i}` (:math:`i = 0, \ldots, N - 1`)
                    :math:`(B, C_{i}, H_{i}, W_{i})`

            Symbols
                :math:`N`
                    Number of tensors in the feature sequence.

        """  # noqa: E501
        return forward_gather(x, self)


class BaseDecoderBlock(nn.Module):
    """Upsampling block followed by a stack of convolutions.

    Subclasses decide how many channels reach ``conv``, which is what
    distinguishes a plain decoder from one that concatenates a skip.
    """

    @typechecked
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        conv_in_channels: int,
        kernel_size: int,
        use_norm: bool,
        align_corners: bool,
        upsample_mode: Literal[
            "simple_upsample", "conv_upsample", "conv_transpose"
        ],
        n_repeats: int,
    ):
        super().__init__()
        self.up = UpBlock(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            use_norm=use_norm,
            align_corners=align_corners,
            upsample_mode=upsample_mode,
        )
        self.conv = ConvStack(
            conv_in_channels, out_channels, n_repeats=n_repeats
        )


class SimpleDecoderBlock(BaseDecoderBlock):
    """Decoder block without a skip connection."""

    @typechecked
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        use_norm: bool,
        align_corners: bool,
        upsample_mode: Literal[
            "simple_upsample", "conv_upsample", "conv_transpose"
        ],
        n_repeats: int,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            conv_in_channels=out_channels,
            kernel_size=kernel_size,
            use_norm=use_norm,
            align_corners=align_corners,
            upsample_mode=upsample_mode,
            n_repeats=n_repeats,
        )

    def forward(self, x: Tensor) -> Tensor:
        r"""Upsample features through one decoder stage.

        Args:
            x: Feature map to upsample through one decoder stage.

        Returns:
            Feature map at the next higher decoder resolution.

        .. shape-contract::

            Inputs
                :math:`x`
                    :math:`(B, C_{\mathrm{in}}, H, W)`

            Outputs
                :math:`\mathrm{output}`
                    :math:`(B, C_{\mathrm{out}}, H_{\mathrm{out}}, W_{\mathrm{out}})`

        """  # noqa: E501
        x = self.up(x)
        return self.conv(x)


class UNetDecoderBlock(BaseDecoderBlock):
    """Decoder block concatenating the matching encoder features."""

    @typechecked
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        use_norm: bool,
        align_corners: bool,
        upsample_mode: Literal[
            "simple_upsample", "conv_upsample", "conv_transpose"
        ],
        n_repeats: int,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            conv_in_channels=in_channels + out_channels,
            kernel_size=kernel_size,
            use_norm=use_norm,
            align_corners=align_corners,
            upsample_mode=upsample_mode,
            n_repeats=n_repeats,
        )

    def forward(self, x: Tensor, skip_x: Tensor) -> Tensor:
        r"""Upsample features and fuse an encoder skip connection.

        Args:
            x: Decoder feature map to upsample.
            skip_x: Encoder feature map for the skip connection.

        Returns:
            Upsampled decoder features fused with the skip connection.

        .. shape-contract::

            Inputs
                :math:`x`
                    :math:`(B, C_{\mathrm{in}}, H, W)`
                :math:`\mathrm{skip}_{\mathrm{x}}`
                    :math:`(B, C_{\mathrm{skip}}, 2 \cdot H, 2 \cdot W)`

            Outputs
                :math:`\mathrm{output}`
                    :math:`(B, C_{\mathrm{out}}, 2 \cdot H, 2 \cdot W)`

        """
        x = self.up(x)
        x = torch.cat([x, skip_x], dim=1)
        return self.conv(x)


class BaseDecoder(nn.Module):
    """Stack of decoder blocks mirroring the encoder widths."""

    @typechecked
    def __init__(
        self,
        base_width: int,
        out_channels: int,
        encoder_width_multipliers: list[float],
        n_convolutions: int,
        block: type[SimpleDecoderBlock | UNetDecoderBlock],
    ):
        super().__init__()
        self.blocks = nn.ModuleList()

        width_multipliers = [1, *encoder_width_multipliers]
        width_multipliers.reverse()
        for i in range(len(width_multipliers) - 1):
            self.blocks.append(
                block(
                    int(width_multipliers[i] * base_width),
                    int(width_multipliers[i + 1] * base_width),
                    kernel_size=3,
                    use_norm=True,
                    align_corners=True,
                    upsample_mode="simple_upsample",
                    n_repeats=n_convolutions,
                )
            )

        self.final_conv = nn.Conv2d(
            base_width, out_channels, kernel_size=3, padding=1
        )


class SimpleDecoder(BaseDecoder):
    """Decoder consuming only the deepest encoder feature map."""

    @typechecked
    def __init__(
        self,
        base_width: int,
        out_channels: int,
        encoder_width_multipliers: list[float],
        n_convolutions: int = 2,
    ):
        super().__init__(
            base_width=base_width,
            out_channels=out_channels,
            encoder_width_multipliers=encoder_width_multipliers,
            n_convolutions=n_convolutions,
            block=SimpleDecoderBlock,
        )

    def forward(self, x: Tensor) -> Tensor:
        r"""Decode a feature map back to image resolution.

        Args:
            x: Compact feature map to decode.

        Returns:
            Decoded feature map at the configured output resolution.

        .. shape-contract::

            Inputs
                :math:`x`
                    :math:`(B, C_{\mathrm{in}}, H, W)`

            Outputs
                :math:`\mathrm{output}`
                    :math:`(B, C_{\mathrm{out}}, H_{\mathrm{out}}, W_{\mathrm{out}})`

        """  # noqa: E501
        for block in self.blocks:
            x = block(x)
        return self.final_conv(x)


class UNetDecoder(BaseDecoder):
    """Decoder consuming the full encoder pyramid through skips."""

    @typechecked
    def __init__(
        self,
        base_width: int,
        out_channels: int,
        encoder_width_multipliers: list[float],
        n_convolutions: int = 2,
    ):
        super().__init__(
            base_width=base_width,
            out_channels=out_channels,
            encoder_width_multipliers=encoder_width_multipliers,
            n_convolutions=n_convolutions,
            block=UNetDecoderBlock,
        )

    def forward(self, inputs: list[Tensor]) -> Tensor:
        r"""Decode a U-Net feature pyramid with skip connections.

        Args:
            inputs: Encoder features ordered from shallowest to deepest.

        Returns:
            Decoded feature map at the highest configured resolution.

        .. shape-contract::

            Inputs
                :math:`\mathrm{inputs}_{i}` (:math:`i = 0, \ldots, N - 1`)
                    :math:`(B, C_{i}, H_{i}, W_{i})`

            Outputs
                :math:`\mathrm{output}`
                    :math:`(B, C_{\mathrm{out}}, H_{\mathrm{out}}, W_{\mathrm{out}})`

            Constraints
                - :math:`N > 0`

            Symbols
                :math:`N`
                    Number of tensors in the feature sequence.

        """  # noqa: E501
        x = inputs.pop()
        for block, skip_x in zip(self.blocks, reversed(inputs), strict=True):
            x = block(x, skip_x)
        return self.final_conv(x)


class UpBlock(nn.Sequential):
    """Upsampling with ConvTranspose2D or Upsample (based on the mode)."""

    @typechecked
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        upsample_mode: Literal[
            "simple_upsample", "conv_upsample", "conv_transpose"
        ],
        kernel_size: int,
        use_norm: bool,
        align_corners: bool,
        stride: int = 2,
        activation: nn.Module | None | bool = True,
        interpolation_mode: Literal[
            "nearest", "linear", "bilinear", "bicubic", "trilinear"
        ] = "bilinear",
    ):
        """Upsampling with ConvTranspose2D or Upsample (based on the
        mode).

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Kernel size. Defaults to ``2``.
            stride: Stride. Defaults to ``2``.
            upsample_mode: Upsampling method, either 'conv_transpose'
                (for ConvTranspose2D) or one of 'simple_upsample' or
                'conv_upsample' (for nn.Upsample). 'conv_upsample' adds
                an additional 1x1 convolution after calling nn.Upsample.
            interpolation_mode: Interpolation mode used for nn.Upsample
                (e.g., 'bilinear', 'nearest').
            align_corners: Align corners option for upsampling methods
                that support it. Defaults to False.
            use_norm: Whether convolutional upsampling uses
                normalization.
            activation: Activation applied after convolutional
                upsampling.

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
            in_channels = out_channels
        else:
            layers.append(
                nn.Upsample(
                    scale_factor=stride,
                    mode=interpolation_mode,
                    align_corners=align_corners,
                )
            )
            if upsample_mode == "conv_upsample":
                layers.append(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1)
                )
                in_channels = out_channels

        layers.append(
            ConvBlock(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                use_norm=use_norm,
                activation=activation,
            )
        )

        super().__init__(*layers)

    def forward(self, x: Tensor) -> Tensor:
        r"""Upsample a feature map by a factor of two.

        Args:
            x: Feature map to upsample.

        Returns:
            Feature map at twice the input spatial resolution.

        .. shape-contract::

            Inputs
                :math:`x`
                    :math:`(B, C_{\mathrm{in}}, H, W)`

            Outputs
                :math:`\mathrm{output}`
                    :math:`(B, C_{\mathrm{out}}, H_{\mathrm{out}}, W_{\mathrm{out}})`

        """  # noqa: E501
        return super().forward(x)
