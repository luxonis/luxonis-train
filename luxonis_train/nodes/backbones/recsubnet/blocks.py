from torch import Tensor, nn

from luxonis_train.nodes.blocks import ConvModule


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvModule(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                activation=nn.ReLU(inplace=True),
            ),
            ConvModule(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                activation=nn.ReLU(inplace=True),
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class Encoder(nn.Module):
    def __init__(self, input_channels: int, width: int) -> None:
        super().__init__()
        self.encoder_block1 = ConvBlock(input_channels, width)
        self.pool1 = nn.MaxPool2d(2)

        self.encoder_block2 = ConvBlock(width, width * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.encoder_block3 = ConvBlock(width * 2, width * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.encoder_block4 = ConvBlock(width * 4, width * 8)
        self.pool4 = nn.MaxPool2d(2)

        self.encoder_block5 = ConvBlock(width * 8, width * 8)

    def forward(self, x: Tensor) -> Tensor:
        enc1 = self.encoder_block1(x)
        enc1_pool = self.pool1(enc1)
        enc2 = self.encoder_block2(enc1_pool)
        enc2_pool = self.pool2(enc2)
        enc3 = self.encoder_block3(enc2_pool)
        enc3_pool = self.pool3(enc3)
        enc4 = self.encoder_block4(enc3_pool)
        enc4_pool = self.pool4(enc4)
        enc5 = self.encoder_block5(enc4_pool)
        return enc5


class Decoder(nn.Module):
    def __init__(self, width: int, out_channels: int = 1) -> None:
        super().__init__()

        self.upscale1 = self.upscale_block(width * 8, width * 8)
        self.decoder_block1 = ConvBlock(width * 8, width * 4)

        self.upscale2 = self.upscale_block(width * 4, width * 4)
        self.decoder_block2 = ConvBlock(width * 4, width * 2)

        self.upscale3 = self.upscale_block(width * 2, width * 2)
        self.decoder_block3 = ConvBlock(width * 2, width)

        self.upscale4 = self.upscale_block(width, width)
        self.decoder_block4 = ConvBlock(width, width)

        self.output_layer = nn.Conv2d(
            width, out_channels, kernel_size=3, padding=1
        )

    def upscale_block(
        self, in_channels: int, out_channels: int
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            ConvModule(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                activation=nn.ReLU(inplace=True),
            ),
        )

    def forward(self, enc5: Tensor) -> Tensor:
        up1 = self.upscale1(enc5)
        dec1 = self.decoder_block1(up1)

        up2 = self.upscale2(dec1)
        dec2 = self.decoder_block2(up2)

        up3 = self.upscale3(dec2)
        dec3 = self.decoder_block3(up3)

        up4 = self.upscale4(dec3)
        dec4 = self.decoder_block4(up4)

        output = self.output_layer(dec4)
        return output


class NanoEncoder(nn.Module):
    def __init__(self, input_channels: int, width: int) -> None:
        super().__init__()

        self.encoder_block1 = ConvBlock(input_channels, width)
        self.pool1 = nn.MaxPool2d(2)

        self.encoder_block2 = ConvBlock(width, int(width * 1.1))
        self.pool2 = nn.MaxPool2d(2)

    def forward(self, x: Tensor) -> Tensor:
        enc1 = self.encoder_block1(x)
        enc1_pool = self.pool1(enc1)
        enc2 = self.encoder_block2(enc1_pool)
        enc2_pool = self.pool2(enc2)
        return enc2_pool


class NanoDecoder(nn.Module):
    def __init__(self, width: int, out_channels: int = 1) -> None:
        super().__init__()

        self.upscale1 = self.upscale_block(int(width * 1.1), width)
        self.decoder_block1 = ConvBlock(width, width)

        self.upscale2 = self.upscale_block(width, width // 2)
        self.decoder_block2 = ConvBlock(width // 2, width // 2)

        self.output_layer = nn.Conv2d(
            width // 2, out_channels, kernel_size=3, padding=1
        )

    def upscale_block(
        self, in_channels: int, out_channels: int
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, enc2: Tensor) -> Tensor:
        up1 = self.upscale1(enc2)
        dec1 = self.decoder_block1(up1)

        up2 = self.upscale2(dec1)
        dec2 = self.decoder_block2(up2)

        output = self.output_layer(dec2)
        return output
