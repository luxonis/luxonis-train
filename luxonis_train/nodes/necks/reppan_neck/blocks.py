from abc import ABC

import torch
from torch import Tensor, nn

from luxonis_train.nodes.blocks import (
    BlockRepeater,
    ConvBlock,
    CSPStackRepBlock,
    GeneralReparametrizableBlock,
)


class PANUpBlockBase(ABC, nn.Module):
    """Base RepPANNeck up block."""

    def __init__(
        self, in_channels: int, out_channels: int, encode_block: nn.Module
    ):
        """Initialize the base upsampling block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            encode_block (nn.Module): Encode block that is used.

        """
        super().__init__()

        self.conv = ConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
        )
        self.upsample = nn.ConvTranspose2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=2,
            stride=2,
            bias=True,
        )
        self.encode_block = encode_block

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

        Args:
            in_channels (int): Number of input channels.
            in_channels_next (int): Number of input channels of next input which is used in concat.
            out_channels (int): Number of output channels.
            n_repeats (int): Number of RepVGGBlock repeats.

        """
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            encode_block=BlockRepeater(
                GeneralReparametrizableBlock,
                in_channels=in_channels_next + out_channels,
                out_channels=out_channels,
                n_repeats=n_repeats,
            ),
        )


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

        Args:
            in_channels (int): Number of input channels.
            in_channels_next (int): Number of input channels of next input which is used in concat.
            out_channels (int): Number of output channels.
            n_repeats (int): Number of RepVGGBlock repeats.
            e (float): Factor that controls number of intermediate channels.

        """
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            encode_block=CSPStackRepBlock(
                in_channels=in_channels_next + out_channels,
                out_channels=out_channels,
                n_blocks=n_repeats,
                e=e,
            ),
        )


class PANDownBlockBase(ABC, nn.Module):
    """Base RepPANNeck down block."""

    def __init__(
        self,
        in_channels: int,
        downsample_out_channels: int,
        encode_block: nn.Module,
    ):
        """Initialize the base downsampling block.

        Args:
            in_channels (int): Number of input channels.
            downsample_out_channels (int): Number of output channels after downsample.
            encode_block (nn.Module): Encode block that is used after concatenation.

        """
        super().__init__()

        self.downsample = ConvBlock(
            in_channels=in_channels,
            out_channels=downsample_out_channels,
            kernel_size=3,
            stride=2,
            padding=3 // 2,
        )
        self.encode_block = encode_block

    def forward(self, x0: Tensor, x1: Tensor) -> Tensor:
        x = self.downsample(x0)
        x = torch.cat([x, x1], dim=1)
        return self.encode_block(x)


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

        Args:
            in_channels (int): Number of input channels.
            downsample_out_channels (int): Number of output channels after downsample.
            in_channels_next (int): Number of input channels of next input which is used in concat.
            out_channels (int): Number of output channels.
            n_repeats (int): Number of RepVGGBlock repeats.

        """
        super().__init__(
            in_channels=in_channels,
            downsample_out_channels=downsample_out_channels,
            encode_block=BlockRepeater(
                GeneralReparametrizableBlock,
                n_repeats=n_repeats,
                in_channels=downsample_out_channels + in_channels_next,
                out_channels=out_channels,
            ),
        )


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

        Args:
            in_channels (int): Number of input channels.
            downsample_out_channels (int): Number of output channels after downsample.
            in_channels_next (int): Number of input channels of next input which is used in concat.
            out_channels (int): Number of output channels.
            n_repeats (int): Number of RepVGGBlock repeats.
            e (float): Factor that controls number of intermediate channels.

        """
        super().__init__(
            in_channels=in_channels,
            downsample_out_channels=downsample_out_channels,
            encode_block=CSPStackRepBlock(
                in_channels=downsample_out_channels + in_channels_next,
                out_channels=out_channels,
                n_blocks=n_repeats,
                e=e,
            ),
        )
