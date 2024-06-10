"""ResNet backbone.

Source: U{https://github.com/hailo-ai/LPRNet_Pytorch/blob/master/model/LPRNet.py}
@license: U{PyTorch<https://github.com/hailo-ai/LPRNet_Pytorch?tab=Apache-2.0-1-ov-file#readme>}
"""
from typing import Literal

import torch
import torch.nn as nn
import torchvision
from torch import Tensor

from .base_node import BaseNode
from luxonis_train.utils.types import LabelType


class ResBlock(nn.Module):
    def __init__(self, ch_in, ch_out, stride=1, ks=3, downsample=None, padding=1):
        super(ResBlock, self).__init__()
        self.downsample = downsample
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=ks, stride=stride, padding=padding),
            nn.BatchNorm2d(num_features=ch_out),
            nn.ReLU(),
            nn.Conv2d(in_channels=ch_out, out_channels=ch_out, kernel_size=ks, stride=1, padding=padding),
            nn.BatchNorm2d(num_features=ch_out),
        )
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.block(x)
        if self.downsample is not None:
            x = self.downsample(x)
        out += x
        out = self.act(out)
        return out


class DownSample(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size=3, stride=1, padding=0):
        super(DownSample, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=kernel_size, stride=stride, padding=padding)
        )

    def forward(self, x):
        out = self.block(x)
        return out


class OCRDecoderBackbone(BaseNode):

    def __init__(
            self,
            num_characters: int = 37,
            in_channels: int = 3,
            dropout_rate: float = 0.5,
            **kwargs
    ):
        super().__init__(**kwargs, _task_type=LabelType.TEXT)
        self.num_characters = num_characters
        self.dropout_rate = dropout_rate

        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),

            ResBlock(ch_in=64, ch_out=64, padding=1),
            ResBlock(ch_in=64, ch_out=128, padding=1,
                     downsample=DownSample(64, 128, kernel_size=1, stride=1)),

            # s2
            ResBlock(ch_in=128, ch_out=128, stride=2, padding=1,
                     downsample=DownSample(128, 128, kernel_size=1, stride=2)),
            ResBlock(ch_in=128, ch_out=256, padding=1,
                     downsample=DownSample(128, 256, kernel_size=1, stride=1)),
        )  # (38 x 150)

        self.downsample1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=2),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=2),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=2),
            nn.BatchNorm2d(num_features=256)
        )

        self.stage2 = nn.Sequential(
            ResBlock(ch_in=256, ch_out=256, stride=2, padding=1,
                     downsample=DownSample(256, 256, kernel_size=1, stride=2)),
            ResBlock(ch_in=256, ch_out=256, padding=1)
        )  # (19 x 75)

        self.downsample2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=2),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=2),
            nn.BatchNorm2d(num_features=256),
        )

        self.stage3 = nn.Sequential(
            ResBlock(ch_in=256, ch_out=256, stride=2, padding=1,
                     downsample=DownSample(256, 256, kernel_size=1, stride=2)),
            ResBlock(ch_in=256, ch_out=256, stride=2, padding=1,
                     downsample=DownSample(256, 256, kernel_size=1, stride=2))
        )  # (5 x 19)
        if dropout_rate > 0:
            self.stage4 = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 5), stride=1, padding=(0, 2)),  # (6 x 24)
                nn.BatchNorm2d(num_features=256),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Conv2d(in_channels=256, out_channels=num_characters, kernel_size=(5, 1), stride=1, padding=(2, 0)),
                # (6 x 24)
                nn.BatchNorm2d(num_features=num_characters),
                nn.ReLU(),
            )
        else:
            self.stage4 = nn.Sequential(
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 5), stride=1, padding=(0, 2)),  # (6 x 24)
                nn.BatchNorm2d(num_features=256),
                nn.ReLU(),
                nn.Conv2d(in_channels=256, out_channels=num_characters, kernel_size=(5, 1), stride=1, padding=(2, 0)),
                # (6 x 24)
                nn.BatchNorm2d(num_features=num_characters),
                nn.ReLU(),
            )  # (5 x 19)

    def forward(self, inputs: Tensor) -> list[Tensor]:
        stage1 = self.stage1(inputs)
        stage2 = self.stage2(stage1)
        stage3 = self.stage3(stage2)
        stage4 = self.stage4(stage3)

        skip1 = self.downsample1(stage1)
        skip2 = self.downsample2(stage2)
        skip3 = stage3
        skip4 = stage4

        return [skip1, skip2, skip3, skip4]


class OCRDecoderHead(BaseNode):

    def __init__(
            self,
            num_characters: int = 37,
            **kwargs
    ):
        super().__init__(**kwargs, _task_type=LabelType.TEXT)

        self.num_characters = num_characters
        self.container = nn.Sequential(
            nn.Conv2d(
                in_channels=768 + self.num_characters,
                out_channels=self.num_characters,
                kernel_size=(1, 1),
                stride=(1, 1)
            )
        )

    def forward(self, inputs: list[Tensor]) -> Tensor:
        features = torch.cat(inputs, dim=1)
        logits = self.container(features)
        logits = torch.mean(logits, dim=2)  # B, Classes, Sequence
        return logits
