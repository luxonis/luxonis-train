from torch import nn


class Encoder(nn.Module):
    def __init__(self, input_channels, width):
        super(Encoder, self).__init__()
        self.encoder_block1 = nn.Sequential(
            nn.Conv2d(input_channels, width, kernel_size=3, padding=1),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, kernel_size=3, padding=1),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(2)

        self.encoder_block2 = nn.Sequential(
            nn.Conv2d(width, width * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(width * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(width * 2, width * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(width * 2),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(2)

        self.encoder_block3 = nn.Sequential(
            nn.Conv2d(width * 2, width * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(width * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(width * 4, width * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(width * 4),
            nn.ReLU(inplace=True),
        )
        self.pool3 = nn.MaxPool2d(2)

        self.encoder_block4 = nn.Sequential(
            nn.Conv2d(width * 4, width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(width * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(width * 8, width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(width * 8),
            nn.ReLU(inplace=True),
        )
        self.pool4 = nn.MaxPool2d(2)

        self.encoder_block5 = nn.Sequential(
            nn.Conv2d(width * 8, width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(width * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(width * 8, width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(width * 8),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
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
    def __init__(self, width, out_channels=1):
        super(Decoder, self).__init__()

        self.upscale1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(width * 8, width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(width * 8),
            nn.ReLU(inplace=True),
        )
        self.decoder_block1 = nn.Sequential(
            nn.Conv2d(width * 8, width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(width * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(width * 8, width * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(width * 4),
            nn.ReLU(inplace=True),
        )

        self.upscale2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(width * 4, width * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(width * 4),
            nn.ReLU(inplace=True),
        )
        self.decoder_block2 = nn.Sequential(
            nn.Conv2d(width * 4, width * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(width * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(width * 4, width * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(width * 2),
            nn.ReLU(inplace=True),
        )

        self.upscale3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(width * 2, width * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(width * 2),
            nn.ReLU(inplace=True),
        )
        self.decoder_block3 = nn.Sequential(
            nn.Conv2d(width * 2, width * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(width * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(width * 2, width, kernel_size=3, padding=1),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )

        self.upscale4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(width, width, kernel_size=3, padding=1),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        self.decoder_block4 = nn.Sequential(
            nn.Conv2d(width, width, kernel_size=3, padding=1),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, kernel_size=3, padding=1),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )

        self.output_layer = nn.Conv2d(
            width, out_channels, kernel_size=3, padding=1
        )

    def forward(self, enc5):
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
