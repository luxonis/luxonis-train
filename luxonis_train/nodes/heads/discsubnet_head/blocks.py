import torch
from torch import Tensor, nn

from luxonis_train.nodes.blocks import ConvModule


def conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
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


def upsample_block(in_channels: int, out_channels: int) -> nn.Sequential:
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


class Encoder(nn.Module):
    def __init__(self, in_channels: int, base_width: int) -> None:
        super().__init__()
        self.enc_block1 = conv_block(in_channels, base_width)
        self.pool1 = nn.MaxPool2d(2)

        self.enc_block2 = conv_block(base_width, base_width * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.enc_block3 = conv_block(base_width * 2, base_width * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.enc_block4 = conv_block(base_width * 4, base_width * 8)
        self.pool4 = nn.MaxPool2d(2)

        self.enc_block5 = conv_block(base_width * 8, base_width * 8)
        self.pool5 = nn.MaxPool2d(2)

        self.enc_block6 = conv_block(base_width * 8, base_width * 8)

    def forward(
        self, x: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        e1 = self.enc_block1(x)
        p1 = self.pool1(e1)
        e2 = self.enc_block2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc_block3(p2)
        p3 = self.pool3(e3)
        e4 = self.enc_block4(p3)
        p4 = self.pool4(e4)
        e5 = self.enc_block5(p4)
        p5 = self.pool5(e5)
        e6 = self.enc_block6(p5)
        return e1, e2, e3, e4, e5, e6


class Decoder(nn.Module):
    def __init__(self, base_width: int, out_channels: int = 1) -> None:
        super().__init__()

        self.up6 = upsample_block(base_width * 8, base_width * 8)
        self.dec_block6 = conv_block(base_width * (8 + 8), base_width * 8)

        self.up5 = upsample_block(base_width * 8, base_width * 4)
        self.dec_block5 = conv_block(base_width * (4 + 8), base_width * 4)

        self.up4 = upsample_block(base_width * 4, base_width * 2)
        self.dec_block4 = conv_block(base_width * (2 + 4), base_width * 2)

        self.up3 = upsample_block(base_width * 2, base_width)
        self.dec_block3 = conv_block(base_width * (2 + 1), base_width)

        self.up2 = upsample_block(base_width, base_width)
        self.dec_block2 = conv_block(base_width * 2, base_width)

        self.output_conv = nn.Conv2d(
            base_width, out_channels, kernel_size=3, padding=1
        )

    def forward(
        self,
        e1: Tensor,
        e2: Tensor,
        e3: Tensor,
        e4: Tensor,
        e5: Tensor,
        e6: Tensor,
    ) -> Tensor:
        up6 = self.up6(e6)
        cat6 = torch.cat((up6, e5), dim=1)
        dec6 = self.dec_block6(cat6)

        up5 = self.up5(dec6)
        cat5 = torch.cat((up5, e4), dim=1)
        dec5 = self.dec_block5(cat5)

        up4 = self.up4(dec5)
        cat4 = torch.cat((up4, e3), dim=1)
        dec4 = self.dec_block4(cat4)

        up3 = self.up3(dec4)
        cat3 = torch.cat((up3, e2), dim=1)
        dec3 = self.dec_block3(cat3)

        up2 = self.up2(dec3)
        cat2 = torch.cat((up2, e1), dim=1)
        dec2 = self.dec_block2(cat2)

        output = self.output_conv(dec2)
        return output


class NanoEncoder(nn.Module):
    def __init__(self, in_channels: int, base_width: int) -> None:
        super().__init__()

        self.enc_block1 = conv_block(in_channels, base_width)
        self.pool1 = nn.MaxPool2d(2)

        self.enc_block2 = conv_block(base_width, int(base_width * 1.1))
        self.pool2 = nn.MaxPool2d(2)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        e1 = self.enc_block1(x)
        p1 = self.pool1(e1)
        e2 = self.enc_block2(p1)
        return e1, e2


class NanoDecoder(nn.Module):
    def __init__(self, base_width: int, out_channels: int = 1) -> None:
        super().__init__()

        self.up2 = upsample_block(int(base_width * 1.1), base_width)
        self.dec_block2 = conv_block(base_width * 2, base_width)

        self.output_conv = nn.Conv2d(
            base_width, out_channels, kernel_size=3, padding=1
        )

    def forward(self, e1: Tensor, e2: Tensor) -> Tensor:
        up2 = self.up2(e2)
        cat2 = torch.cat((up2, e1), dim=1)
        dec2 = self.dec_block2(cat2)

        output = self.output_conv(dec2)
        return output
