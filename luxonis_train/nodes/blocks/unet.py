from typing import Literal

import torch
from torch import Tensor, nn
from typeguard import typechecked

from .blocks import ConvBlock, ConvStack
from .utils import forward_gather


class EncoderBlock(nn.Sequential):
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


class SimpleEncoder(nn.Sequential):
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


class UNetEncoder(SimpleEncoder):
    def forward(self, x: Tensor) -> list[Tensor]:
        return forward_gather(x, self)


class BaseDecoderBlock(nn.Module):
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
        x = self.up(x)
        return self.conv(x)


class UNetDecoderBlock(BaseDecoderBlock):
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
        x = self.up(x)
        x = torch.cat([x, skip_x], dim=1)
        return self.conv(x)


class BaseDecoder(nn.Module):
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
        for block in self.blocks:
            x = block(x)
        return self.final_conv(x)


class UNetDecoder(BaseDecoder):
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
        x = inputs.pop()
        for block, skip_x in zip(self.blocks, reversed(inputs), strict=True):
            x = block(x, skip_x)
        return self.final_conv(x)


class UpBlock(nn.Sequential):
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

        @type in_channels: int
        @param in_channels: Number of input channels.
        @type out_channels: int
        @param out_channels: Number of output channels.
        @type kernel_size: int
        @param kernel_size: Kernel size. Defaults to C{2}.
        @type stride: int
        @param stride: Stride. Defaults to C{2}.
        @type upsample_mode: Literal["simple_upsample", "conv_upsample",
            "conv_transpose"]
        @param upsample_mode: Upsampling method, either 'conv_transpose'
            (for ConvTranspose2D) or one of 'simple_upsample' or
            'conv_upsample' (for nn.Upsample). 'conv_upsample' adds an
            additional 1x1 convolution after calling nn.Upsample.
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
