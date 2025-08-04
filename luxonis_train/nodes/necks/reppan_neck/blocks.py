from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn

from luxonis_train.nodes.blocks import (
    BlockRepeater,
    ConvModule,
    CSPStackRepBlock,
    RepVGGBlock,
)


class PANUpBlockBase(ABC, nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        """Base RepPANNeck up block.

        @type in_channels: int
        @param in_channels: Number of input channels.
        @type out_channels: int
        @param out_channels: Number of output channels.
        """
        super().__init__()

        self.conv = ConvModule(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
        )
        self.upsample = torch.nn.ConvTranspose2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=2,
            stride=2,
            bias=True,
        )

    @property
    @abstractmethod
    def encode_block(self) -> nn.Module:
        """Encode block that is used.

        Make sure actual module is initialized in the __init__ and not
        inside this function otherwise it will be reinitialized every
        time
        """
        ...

    def forward(self, x0: Tensor, x1: Tensor) -> tuple[Tensor, Tensor]:
        conv_out = self.conv(x0)
        upsample_out = self.upsample(conv_out)
        concat_out = torch.cat([upsample_out, x1], dim=1)
        out = self.encode_block(concat_out)
        return conv_out, out


class RepUpBlock(PANUpBlockBase):
    def __init__(
        self,
        in_channels: int,
        in_channels_next: int,
        out_channels: int,
        n_repeats: int,
    ):
        """RepPANNeck up block for smaller networks that uses RepBlock.

        @type in_channels: int
        @param in_channels: Number of input channels.
        @type in_channels_next: int
        @param in_channels_next: Number of input channels of next input
            which is used in concat.
        @type out_channels: int
        @param out_channels: Number of output channels.
        @type n_repeats: int
        @param n_repeats: Number of RepVGGBlock repeats.
        """
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
        )

        self._encode_block = BlockRepeater(
            block=RepVGGBlock,
            in_channels=in_channels_next + out_channels,
            out_channels=out_channels,
            n_blocks=n_repeats,
        )

    @property
    def encode_block(self) -> nn.Module:
        return self._encode_block


class CSPUpBlock(PANUpBlockBase):
    def __init__(
        self,
        in_channels: int,
        in_channels_next: int,
        out_channels: int,
        n_repeats: int,
        e: float,
    ):
        """RepPANNeck up block for larger networks that uses
        CSPStackRepBlock.

        @type in_channels: int
        @param in_channels: Number of input channels.
        @type in_channels_next: int
        @param in_channels_next: Number of input channels of next input
            which is used in concat.
        @type out_channels: int
        @param out_channels: Number of output channels.
        @type n_repeats: int
        @param n_repeats: Number of RepVGGBlock repeats.
        @type e: float
        @param e: Factor that controls number of intermediate channels.
        """
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
        )
        self._encode_block = CSPStackRepBlock(
            in_channels=in_channels_next + out_channels,
            out_channels=out_channels,
            n_blocks=n_repeats,
            e=e,
        )

    @property
    def encode_block(self) -> nn.Module:
        return self._encode_block


class PANDownBlockBase(ABC, nn.Module):
    def __init__(
        self,
        in_channels: int,
        downsample_out_channels: int,
    ):
        """Base RepPANNeck up block.

        @type in_channels: int
        @param in_channels: Number of input channels.
        @type downsample_out_channels: int
        @param downsample_out_channels: Number of output channels after
            downsample.
        @type in_channels_next: int
        @param in_channels_next: Number of input channels of next input
            which is used in concat.
        @type out_channels: int
        @param out_channels: Number of output channels.
        @type n_repeats: int
        @param n_repeats: Number of RepVGGBlock repeats.
        """
        super().__init__()

        self.downsample = ConvModule(
            in_channels=in_channels,
            out_channels=downsample_out_channels,
            kernel_size=3,
            stride=2,
            padding=3 // 2,
        )

    @property
    @abstractmethod
    def encode_block(self) -> nn.Module:
        """Encode block that is used.

        Make sure actual module is initialized in the __init__ and not
        inside this function otherwise it will be reinitialized every
        time
        """
        ...

    def forward(self, x0: Tensor, x1: Tensor) -> Tensor:
        x = self.downsample(x0)
        x = torch.cat([x, x1], dim=1)
        x = self.encode_block(x)
        return x


class RepDownBlock(PANDownBlockBase):
    def __init__(
        self,
        in_channels: int,
        downsample_out_channels: int,
        in_channels_next: int,
        out_channels: int,
        n_repeats: int,
    ):
        """RepPANNeck down block for smaller networks that uses
        RepBlock.

        @type in_channels: int
        @param in_channels: Number of input channels.
        @type downsample_out_channels: int
        @param downsample_out_channels: Number of output channels after
            downsample.
        @type in_channels_next: int
        @param in_channels_next: Number of input channels of next input
            which is used in concat.
        @type out_channels: int
        @param out_channels: Number of output channels.
        @type n_repeats: int
        @param n_repeats: Number of RepVGGBlock repeats.
        """
        super().__init__(
            in_channels=in_channels,
            downsample_out_channels=downsample_out_channels,
        )

        self._encode_block = BlockRepeater(
            block=RepVGGBlock,
            in_channels=downsample_out_channels + in_channels_next,
            out_channels=out_channels,
            n_blocks=n_repeats,
        )

    @property
    def encode_block(self) -> nn.Module:
        return self._encode_block


class CSPDownBlock(PANDownBlockBase):
    def __init__(
        self,
        in_channels: int,
        downsample_out_channels: int,
        in_channels_next: int,
        out_channels: int,
        n_repeats: int,
        e: float,
    ):
        """RepPANNeck up block for larger networks that uses
        CSPStackRepBlock.

        @type in_channels: int
        @param in_channels: Number of input channels.
        @type downsample_out_channels: int
        @param downsample_out_channels: Number of output channels after
            downsample.
        @type in_channels_next: int
        @param in_channels_next: Number of input channels of next input
            which is used in concat.
        @type out_channels: int
        @param out_channels: Number of output channels.
        @type n_repeats: int
        @param n_repeats: Number of RepVGGBlock repeats.
        @type e: float
        @param e: Factor that controls number of intermediate channels.
        """
        super().__init__(
            in_channels=in_channels,
            downsample_out_channels=downsample_out_channels,
        )

        self._encode_block = CSPStackRepBlock(
            in_channels=downsample_out_channels + in_channels_next,
            out_channels=out_channels,
            n_blocks=n_repeats,
            e=e,
        )

    @property
    def encode_block(self) -> nn.Module:
        return self._encode_block
