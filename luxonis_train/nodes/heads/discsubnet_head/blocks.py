import torch
from torch import Tensor, nn


class Encoder(nn.Module):
    def __init__(self, in_channels: int, base_width: int) -> None:
        super(Encoder, self).__init__()
        self.enc_block1 = nn.Sequential(
            nn.Conv2d(in_channels, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(2)

        self.enc_block2 = nn.Sequential(
            nn.Conv2d(base_width, base_width * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                base_width * 2, base_width * 2, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(2)

        self.enc_block3 = nn.Sequential(
            nn.Conv2d(
                base_width * 2, base_width * 4, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                base_width * 4, base_width * 4, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True),
        )
        self.pool3 = nn.MaxPool2d(2)

        self.enc_block4 = nn.Sequential(
            nn.Conv2d(
                base_width * 4, base_width * 8, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                base_width * 8, base_width * 8, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
        )
        self.pool4 = nn.MaxPool2d(2)

        self.enc_block5 = nn.Sequential(
            nn.Conv2d(
                base_width * 8, base_width * 8, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                base_width * 8, base_width * 8, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
        )

        self.pool5 = nn.MaxPool2d(2)

        self.enc_block6 = nn.Sequential(
            nn.Conv2d(
                base_width * 8, base_width * 8, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                base_width * 8, base_width * 8, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
        )

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
        super(Decoder, self).__init__()

        self.up6 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(
                base_width * 8, base_width * 8, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
        )
        self.dec_block6 = nn.Sequential(
            nn.Conv2d(
                base_width * (8 + 8), base_width * 8, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                base_width * 8, base_width * 8, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
        )

        self.up5 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(
                base_width * 8, base_width * 4, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True),
        )
        self.dec_block5 = nn.Sequential(
            nn.Conv2d(
                base_width * (4 + 8), base_width * 4, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                base_width * 4, base_width * 4, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True),
        )

        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(
                base_width * 4, base_width * 2, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True),
        )
        self.dec_block4 = nn.Sequential(
            nn.Conv2d(
                base_width * (2 + 4), base_width * 2, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                base_width * 2, base_width * 2, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True),
        )

        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(base_width * 2, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
        )
        self.dec_block3 = nn.Sequential(
            nn.Conv2d(
                base_width * (2 + 1), base_width, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
        )

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
        )
        self.dec_block2 = nn.Sequential(
            nn.Conv2d(base_width * 2, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
        )

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
