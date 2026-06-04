"""DDRNet blocks.

Adapted from: `https://github.com/Deci-AI/super-gradients/blob/master/src/super_gradients/training/models/segmentation_models/ddrnet.py <https://github.com/Deci-AI/super-gradients/blob/master/src/super_gradients/training/models/segmentation_models/ddrnet.py>`_
Original source: `https://github.com/ydhongHIT/DDRNet <https://github.com/ydhongHIT/DDRNet>`_
Paper: `https://arxiv.org/pdf/2101.06085.pdf <https://arxiv.org/pdf/2101.06085.pdf>`_

Notes:
    License: `https://github.com/Deci-AI/super-gradients/blob/master/LICENSE.md <https://github.com/Deci-AI/super-gradients/blob/master/LICENSE.md>`_

"""

import torch
from torch import Tensor, nn

from luxonis_train.nodes.blocks import ConvBlock, UpscaleOnline


class DAPPMBranch(nn.Module):
    """Branch of the DAPPM module."""

    def __init__(
        self,
        in_channels: int,
        kernel_size: int,
        stride: int,
        branch_channels: int,
        interpolation_mode: str = "bilinear",
    ):
        """Initialize a DAPPM branch.

        Args:
            in_channels (int): Number of input channels.
            kernel_size (int): The kernel size. When stride=0, this parameter is omitted, and AdaptiveAvgPool2d over all the input is performed.
            stride (int): Stride for the first convolution. When stride is set to 0, ``AdaptiveAvgPool2d`` over all the input is performed (output is 1x1). When set to 1, no operation is performed. When stride>1, a convolution with ``stride=stride`` is performed.
            branch_channels (int): Width after the first convolution.
            interpolation_mode (str): Interpolation mode for upscaling. Defaults to "bilinear".

        """
        super().__init__()

        down_list = []
        down_list.append(nn.BatchNorm2d(in_channels))
        if stride == 0:
            down_list.append(nn.AdaptiveAvgPool2d((1, 1)))
        elif stride > 1:
            down_list.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=stride,
                    groups=in_channels,
                    bias=False,
                )
            )

        down_list.append(nn.ReLU(inplace=True))
        down_list.append(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, bias=False)
        )

        self.down_scale = nn.Sequential(*down_list)
        self.up_scale = UpscaleOnline(interpolation_mode)

    def forward(self, x: Tensor) -> Tensor:
        h, w = x.shape[-2], x.shape[-1]
        out = self.down_scale(x)
        return self.up_scale(out, output_height=h, output_width=w)


class MergeDAPPMBranch(DAPPMBranch):
    """A DAPPM branch working with an input from the previous branch."""

    def __init__(
        self,
        in_channels: int,
        kernel_size: int,
        stride: int,
        branch_channels: int,
        interpolation_mode: str = "bilinear",
    ):
        """Initialize a merge DAPPM branch.

        Args:
            in_channels (int): Number of input channels.
            kernel_size (int): The kernel size. When stride=0, this parameter is omitted, and AdaptiveAvgPool2d over all the input is performed.
            stride (int): Stride for the first convolution. When stride is set to 0, ``AdaptiveAvgPool2d`` over all the input is performed (output is 1x1). When set to 1, no operation is performed. When stride>1, a convolution with ``stride=stride`` is performed.
            branch_channels (int): Width after the first convolution.
            interpolation_mode (str): Interpolation mode for upscaling. Defaults to "bilinear".

        """
        super().__init__(
            kernel_size=kernel_size,
            stride=stride,
            in_channels=in_channels,
            branch_channels=branch_channels,
            interpolation_mode=interpolation_mode,
        )

        self.process = nn.Sequential(
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                branch_channels,
                branch_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
        )

    def forward(self, x: Tensor, skip_x: Tensor) -> Tensor:
        out = super().forward(x)
        return self.process(out + skip_x)


class DAPPM(nn.Module):
    def __init__(
        self,
        in_channels: int,
        branch_channels: int,
        out_channels: int,
        kernel_sizes: list[int],
        strides: list[int],
        interpolation_mode: str = "bilinear",
    ):
        """DAPPM (Dynamic Attention Pyramid Pooling Module).

        Args:
            in_channels (int): Number of input channels.
            branch_channels (int): Width after the first convolution in each branch.
            out_channels (int): Number of output channels.
            kernel_sizes (list[int]): List of kernel sizes for each branch.
            strides (list[int]): List of strides for each branch.
            interpolation_mode (str): Interpolation mode for upscaling. Defaults to "bilinear".

        Raises:
            ValueError: If the lengths of ``kernel_sizes`` and ``strides`` are not the same.

        """
        super().__init__()

        self.start_branch = DAPPMBranch(
            in_channels=in_channels,
            kernel_size=kernel_sizes[0],
            stride=strides[0],
            branch_channels=branch_channels,
            interpolation_mode=interpolation_mode,
        )
        self.branches = nn.ModuleList(
            [
                MergeDAPPMBranch(
                    in_channels=in_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    branch_channels=branch_channels,
                    interpolation_mode=interpolation_mode,
                )
                for kernel_size, stride in zip(
                    kernel_sizes[1:], strides[1:], strict=True
                )
            ]
        )

        compression_channels = branch_channels * (len(self.branches) + 1)
        self.compression = nn.Sequential(
            nn.BatchNorm2d(compression_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                compression_channels,
                out_channels,
                kernel_size=1,
                bias=False,
            ),
        )
        self.shortcut = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the DAPPM module.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after processing through all branches and compression.

        """
        x_list = [self.start_branch(x)]

        for i, branch in enumerate(self.branches):
            x_list.append(branch(x, x_list[i]))

        return self.compression(torch.cat(x_list, dim=1)) + self.shortcut(x)


class BasicDDRBackbone(nn.Module):
    def __init__(
        self,
        block: type[nn.Module],
        stem_channels: int,
        layers: list[int],
        in_channels: int,
        layer3_repeats: int = 1,
    ):
        """Initialize the BasicDDRBackBone with specified parameters.

        Args:
            block (Type[nn.Module]): The block class to use for layers.
            stem_channels (int): Number of output channels in the stem layer.
            layers (list[int]): Number of blocks in each layer.
            in_channels (int): Number of input channels.
            layer3_repeats (int): Number of repeats for layer3. Defaults to 1.

        """
        super().__init__()
        self.input_channels = in_channels

        self.stem = nn.Sequential(
            ConvBlock(
                in_channels=in_channels,
                out_channels=stem_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=True,
            ),
            ConvBlock(
                in_channels=stem_channels,
                out_channels=stem_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=True,
            ),
        )

        self.layer1 = make_layer(
            block=block,
            in_channels=stem_channels,
            channels=stem_channels,
            n_blocks=layers[0],
        )

        self.layer2 = make_layer(
            block=block,
            in_channels=stem_channels,
            channels=stem_channels * 2,
            n_blocks=layers[1],
            stride=2,
        )

        self.layer3 = nn.ModuleList(
            [
                make_layer(
                    block=block,
                    in_channels=stem_channels * 2,
                    channels=stem_channels * 4,
                    n_blocks=layers[2],
                    stride=2,
                )
            ]
            + [
                make_layer(
                    block=block,
                    in_channels=stem_channels * 4,
                    channels=stem_channels * 4,
                    n_blocks=layers[2],
                    stride=1,
                )
                for _ in range(layer3_repeats - 1)
            ]
        )

        self.layer4 = make_layer(
            block=block,
            in_channels=stem_channels * 4,
            channels=stem_channels * 8,
            n_blocks=layers[3],
            stride=2,
        )

    def get_backbone_output_number_of_channels(self) -> dict[str, int]:
        """Determine the number of output channels for each backbone
        layer.

        Returns a dictionary with keys "layer2", "layer3", "layer4" and
        their respective number of output channels.

        Returns:
            dict[str, int]: Dictionary of output channel counts for each layer.

        """
        output_shapes = {}
        x = torch.randn(1, self.input_channels, 320, 320)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        output_shapes["layer2"] = x.shape[1]

        for layer in self.layer3:
            x = layer(x)
        output_shapes["layer3"] = x.shape[1]

        x = self.layer4(x)
        output_shapes["layer4"] = x.shape[1]

        return output_shapes


def make_layer(
    block: type[nn.Module],
    in_channels: int,
    channels: int,
    n_blocks: int,
    stride: int = 1,
    expansion: int = 1,
) -> nn.Sequential:
    """Create a sequential layer consisting of a series of blocks.

    Args:
        block (type[nn.Module]): The block class to be used.
        in_channels (int): Number of input channels.
        channels (int): Number of output channels.
        n_blocks (int): Number of blocks in the layer.
        stride (int): Stride for the first block. Defaults to 1.
        expansion (int): Expansion factor for the block. Defaults to 1.

    Returns:
        nn.Sequential: A sequential container of the blocks.

    """
    layers: list[nn.Module] = []

    layers.append(
        block(
            in_channels,
            channels,
            stride=stride,
            final_relu=n_blocks > 1,
            expansion=expansion,
        )
    )

    in_channels = channels * expansion

    for i in range(1, n_blocks):
        final_relu = i != (n_blocks - 1)
        layers.append(
            block(
                in_channels,
                channels,
                stride=1,
                final_relu=final_relu,
                expansion=expansion,
            )
        )

    return nn.Sequential(*layers)
