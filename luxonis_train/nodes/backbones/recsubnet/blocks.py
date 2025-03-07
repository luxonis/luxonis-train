from torch import nn

from luxonis_train.nodes.blocks import ConvModule
from luxonis_train.nodes.blocks.blocks import ModuleRepeater


class ConvStack(ModuleRepeater):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            ConvModule,
            n_repeats=2,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
        )


class Encoder(nn.Sequential):
    def __init__(self, input_channels: int, width: int) -> None:
        super().__init__(
            ConvStack(input_channels, width),
            nn.MaxPool2d(2),
            ConvStack(width, width * 2),
            nn.MaxPool2d(2),
            ConvStack(width * 2, width * 4),
            nn.MaxPool2d(2),
            ConvStack(width * 4, width * 8),
            nn.MaxPool2d(2),
            ConvStack(width * 8, width * 8),
        )


class Decoder(nn.Module):
    def __init__(self, width: int, out_channels: int = 1) -> None:
        super().__init__(
            *self.upscale_block(width * 8, width * 8),
            ConvStack(width * 8, width * 4),
            *self.upscale_block(width * 4, width * 4),
            ConvStack(width * 4, width * 2),
            *self.upscale_block(width * 2, width * 2),
            ConvStack(width * 2, width),
            *self.upscale_block(width, width),
            ConvStack(width, width),
            nn.Conv2d(width, out_channels, kernel_size=3, padding=1),
        )

    def upscale_block(
        self, in_channels: int, out_channels: int
    ) -> list[nn.Module]:
        return [
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            ConvModule(in_channels, out_channels, kernel_size=3, padding=1),
        ]


class NanoEncoder(nn.Sequential):
    def __init__(self, input_channels: int, width: int) -> None:
        super().__init__(
            ConvStack(input_channels, width),
            nn.MaxPool2d(2),
            ConvStack(width, int(width * 1.1)),
            nn.MaxPool2d(2),
        )


class NanoDecoder(nn.Sequential):
    def __init__(self, width: int, out_channels: int = 1) -> None:
        super().__init__(
            *self.upscale_block(int(width * 1.1), width),
            ConvStack(width, width),
            *self.upscale_block(width, width // 2),
            ConvStack(width // 2, width // 2),
            nn.Conv2d(width // 2, out_channels, kernel_size=3, padding=1),
        )

    def upscale_block(
        self, in_channels: int, out_channels: int
    ) -> list[nn.Module]:
        return [
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            ConvModule(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                use_norm=False,
            ),
        ]
