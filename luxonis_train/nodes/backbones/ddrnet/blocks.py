"""DDRNet blocks.

Adapted from: U{https://github.com/Deci-AI/super-gradients/blob/master/src/super_gradients/training/models/segmentation_models/ddrnet.py}
Original source: U{https://github.com/ydhongHIT/DDRNet}
Paper: U{https://arxiv.org/pdf/2101.06085.pdf}
@license: U{https://github.com/Deci-AI/super-gradients/blob/master/LICENSE.md}
"""
from abc import ABC
from typing import Dict, Type

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from luxonis_train.nodes.base_node import BaseNode
from luxonis_train.nodes.blocks import ConvModule


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample, when applied in the main path of
    residual blocks.

    Intended usage of this block is as follows:

    >>> class ResNetBlock(nn.Module):
    >>>   def __init__(self, ..., drop_path_rate: float):
    >>>     self.drop_path = DropPath(drop_path_rate)
    >>>
    >>>   def forward(self, x):
    >>>     return x + self.drop_path(self.conv_bn_act(x))

    Code taken from TIMM (https://github.com/rwightman/pytorch-image-models), Apache License 2.0.
    """

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        """Initializes the DropPath module.

        @type drop_prob: float
        @param drop_prob: Probability of zeroing out individual vectors (channel
            dimension) of each feature map. Defaults to 0.0.
        @type scale_by_keep: bool
        @param scale_by_keep: Whether to scale the output by the keep probability.
            Enabled by default to maintain output mean & std in the same range as
            without DropPath. Defaults to True.
        """
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        return drop_path(x, self.drop_prob, self.scale_by_keep)

    def extra_repr(self) -> str:
        return f"drop_prob={round(self.drop_prob, 3):0.3f}"


class BasicResNetBlock(nn.Module):
    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        expansion: int = 1,
        final_relu: bool = True,
        droppath_prob: float = 0.0,
    ):
        """A basic residual block for ResNet.

        @type in_planes: int
        @param in_planes: Number of input channels.
        @type planes: int
        @param planes: Number of output channels.
        @type stride: int
        @param stride: Stride for the convolutional layers. Defaults to 1.
        @type expansion: int
        @param expansion: Expansion factor for the output channels. Defaults to 1.
        @type final_relu: bool
        @param final_relu: Whether to apply a ReLU activation after the residual
            addition. Defaults to True.
        @type droppath_prob: float
        @param droppath_prob: Drop path probability for stochastic depth. Defaults to
            0.0.
        """
        super().__init__()
        self.expansion = expansion
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.final_relu = final_relu

        self.drop_path = DropPath(drop_prob=droppath_prob)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x: Tensor) -> Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.drop_path(out)
        out += self.shortcut(x)
        if self.final_relu:
            out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        expansion: int = 4,
        final_relu: bool = True,
        droppath_prob: float = 0.0,
    ):
        """A bottleneck block for ResNet.

        @type in_planes: int
        @param in_planes: Number of input channels.
        @type planes: int
        @param planes: Number of intermediate channels.
        @type stride: int
        @param stride: Stride for the second convolutional layer. Defaults to 1.
        @type expansion: int
        @param expansion: Expansion factor for the output channels. Defaults to 4.
        @type final_relu: bool
        @param final_relu: Whether to apply a ReLU activation after the residual
            addition. Defaults to True.
        @type droppath_prob: float
        @param droppath_prob: Drop path probability for stochastic depth. Defaults to
            0.0.
        """
        super().__init__()
        self.expansion = expansion
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.final_relu = final_relu

        self.drop_path = DropPath(drop_prob=droppath_prob)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x: Tensor) -> Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        out = self.drop_path(out)
        out += self.shortcut(x)

        if self.final_relu:
            out = F.relu(out)

        return out


class DAPPMBranch(nn.Module):
    def __init__(
        self,
        kernel_size: int,
        stride: int,
        in_planes: int,
        branch_planes: int,
        inter_mode: str = "bilinear",
    ):
        """A DAPPM branch.

        @type kernel_size: int
        @param kernel_size: The kernel size for the average pooling. When stride=0, this
            parameter is omitted, and AdaptiveAvgPool2d over all the input is performed.
        @type stride: int
        @param stride: Stride for the average pooling. When stride=0, an
            AdaptiveAvgPool2d over all the input is performed (output is 1x1). When
            stride=1, no average pooling is performed. When stride>1, average pooling is
            performed (scaling the input down and up again).
        @type in_planes: int
        @param in_planes: Number of input channels.
        @type branch_planes: int
        @param branch_planes: Width after the first convolution.
        @type inter_mode: str
        @param inter_mode: Interpolation mode for upscaling. Defaults to "bilinear".
        """
        super().__init__()

        down_list = []
        if stride == 0:
            down_list.append(nn.AdaptiveAvgPool2d((1, 1)))
        elif stride > 1:
            down_list.append(
                nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=stride)
            )

        down_list.append(nn.BatchNorm2d(in_planes))
        down_list.append(nn.ReLU(inplace=True))
        down_list.append(nn.Conv2d(in_planes, branch_planes, kernel_size=1, bias=False))

        self.down_scale = nn.Sequential(*down_list)
        self.up_scale = UpscaleOnline(inter_mode)

        if stride != 1:
            self.process = nn.Sequential(
                nn.BatchNorm2d(branch_planes),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    branch_planes, branch_planes, kernel_size=3, padding=1, bias=False
                ),
            )

    def forward(self, x: Tensor) -> Tensor:
        """Process input through the DAPPM branch.

        @type x: Tensor or list[Tensor]
        @param x: In branch 0 - the original input of the DAPPM. In other branches - a list containing the original
                  input and the output of the previous branch.

        @return: Processed output tensor.
        """
        if isinstance(x, list):
            output_of_prev_branch = x[1]
            x = x[0]
        else:
            output_of_prev_branch = None

        in_width = x.shape[-1]
        in_height = x.shape[-2]
        out = self.down_scale(x)
        out = self.up_scale(out, output_height=in_height, output_width=in_width)

        if output_of_prev_branch is not None:
            out = self.process(out + output_of_prev_branch)

        return out


class DAPPM(nn.Module):
    def __init__(
        self,
        in_planes: int,
        branch_planes: int,
        out_planes: int,
        kernel_sizes: list[int],
        strides: list[int],
        inter_mode: str = "bilinear",
    ):
        """DAPPM (Dynamic Attention Pyramid Pooling Module).

        @type in_planes: int
        @param in_planes: Number of input channels.
        @type branch_planes: int
        @param branch_planes: Width after the first convolution in each branch.
        @type out_planes: int
        @param out_planes: Number of output channels.
        @type kernel_sizes: list[int]
        @param kernel_sizes: List of kernel sizes for each branch.
        @type strides: list[int]
        @param strides: List of strides for each branch.
        @type inter_mode: str
        @param inter_mode: Interpolation mode for upscaling. Defaults to "bilinear".
        """
        super().__init__()

        assert len(kernel_sizes) == len(
            strides
        ), "len of kernel_sizes and strides must be the same"

        self.branches = nn.ModuleList(
            [
                DAPPMBranch(
                    kernel_size=kernel_size,
                    stride=stride,
                    in_planes=in_planes,
                    branch_planes=branch_planes,
                    inter_mode=inter_mode,
                )
                for kernel_size, stride in zip(kernel_sizes, strides)
            ]
        )

        self.compression = nn.Sequential(
            nn.BatchNorm2d(branch_planes * len(self.branches)),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                branch_planes * len(self.branches),
                out_planes,
                kernel_size=1,
                bias=False,
            ),
        )
        self.shortcut = nn.Sequential(
            nn.BatchNorm2d(in_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the DAPPM module.

        @type x: Tensor
        @param x: Input tensor.
        @return: Output tensor after processing through all branches and compression.
        """
        x_list = [self.branches[0](x)]

        for i in range(1, len(self.branches)):
            x_list.append(self.branches[i]([x, x_list[i - 1]]))

        out = self.compression(torch.cat(x_list, dim=1)) + self.shortcut(x)
        return out


class UpscaleOnline(nn.Module):
    """Upscale tensor to a specified size during the forward pass.

    This class supports cases where the required scale/size is only known when the input
    is received. Only the interpolation mode is set in advance.
    """

    def __init__(self, mode: str = "bilinear"):
        """Initialize UpscaleOnline with the interpolation mode.

        @type mode: str
        @param mode: Interpolation mode for resizing. Defaults to "bilinear".
        """
        super().__init__()
        self.mode = mode

    def forward(self, x: Tensor, output_height: int, output_width: int) -> Tensor:
        """Upscale the input tensor to the specified height and width.

        @type x: Tensor
        @param x: Input tensor to be upscaled.
        @type output_height: int
        @param output_height: Desired height of the output tensor.
        @type output_width: int
        @param output_width: Desired width of the output tensor.
        @return: Upscaled tensor.
        """
        return F.interpolate(x, size=[output_height, output_width], mode=self.mode)

class BasicDDRBackBone(nn.Module):
    def __init__(
        self,
        block: Type[nn.Module],
        width: int,
        layers: list[int],
        input_channels: int,
        layer3_repeats: int = 1,
    ):
        """Initialize the BasicDDRBackBone with specified parameters.

        @type block: Type[nn.Module]
        @param block: The block class to use for layers.
        @type width: int
        @param width: Width of the feature maps.
        @type layers: list[int]
        @param layers: Number of blocks in each layer.
        @type input_channels: int
        @param input_channels: Number of input channels.
        @type layer3_repeats: int
        @param layer3_repeats: Number of repeats for layer3. Defaults to 1.
        """
        super().__init__()
        self.input_channels = input_channels

        self.stem = nn.Sequential(
            ConvModule(
                in_channels=input_channels,
                out_channels=width,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=True,
                activation=nn.ReLU(inplace=True),
            ),
            ConvModule(
                in_channels=width,
                out_channels=width,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=True,
                activation=nn.ReLU(inplace=True),
            ),
        )

        self.layer1 = _make_layer(
            block=block,
            in_planes=width,
            planes=width,
            num_blocks=layers[0],
        )

        self.layer2 = _make_layer(
            block=block,
            in_planes=width,
            planes=width * 2,
            num_blocks=layers[1],
            stride=2,
        )

        self.layer3 = nn.ModuleList(
            [
                _make_layer(
                    block=block,
                    in_planes=width * 2,
                    planes=width * 4,
                    num_blocks=layers[2],
                    stride=2,
                )
            ]
            + [
                _make_layer(
                    block=block,
                    in_planes=width * 4,
                    planes=width * 4,
                    num_blocks=layers[2],
                    stride=1,
                )
                for _ in range(layer3_repeats - 1)
            ]
        )

        self.layer4 = _make_layer(
            block=block,
            in_planes=width * 4,
            planes=width * 8,
            num_blocks=layers[3],
            stride=2,
        )

    def validate_backbone_attributes(self) -> None:
        """Validate the existence of required backbone attributes.

        Ensures that the following attributes are present: "stem", "layer1", "layer2",
        "layer3", "layer4", "input_channels".
        """
        expected_attributes = [
            "stem",
            "layer1",
            "layer2",
            "layer3",
            "layer4",
            "input_channels",
        ]
        for attribute in expected_attributes:
            assert hasattr(
                self, attribute
            ), f"Invalid backbone - attribute '{attribute}' is missing"

    def get_backbone_output_number_of_channels(self) -> dict[str, int]:
        """Determine the number of output channels for each layer of the backbone.

        Returns a dictionary with keys "layer2", "layer3", "layer4" and their respective
        number of output channels.

        @return: Dictionary of output channel counts for each layer.
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

def _make_layer(
    block: Type[nn.Module],
    in_planes: int,
    planes: int,
    num_blocks: int,
    stride: int = 1,
    expansion: int = 1,
) -> nn.Sequential:
    """Creates a sequential layer consisting of a series of blocks.

    @type block: Type[nn.Module]
    @param block: The block class to be used.
    @type in_planes: int
    @param in_planes: Number of input channels.
    @type planes: int
    @param planes: Number of output channels.
    @type num_blocks: int
    @param num_blocks: Number of blocks in the layer.
    @type stride: int
    @param stride: Stride for the first block. Defaults to 1.
    @type expansion: int
    @param expansion: Expansion factor for the block. Defaults to 1.
    @return: A sequential container of the blocks.
    """
    layers: list[nn.Module] = []

    layers.append(
        block(in_planes, planes, stride, final_relu=num_blocks > 1, expansion=expansion)
    )

    in_planes = planes * expansion

    if num_blocks > 1:
        for i in range(1, num_blocks):
            final_relu = i != (num_blocks - 1)
            layers.append(
                block(
                    in_planes,
                    planes,
                    stride=1,
                    final_relu=final_relu,
                    expansion=expansion,
                )
            )

    return nn.Sequential(*layers)


def drop_path(x: Tensor, drop_prob: float = 0.0, scale_by_keep: bool = True) -> Tensor:
    """Drop paths (Stochastic Depth) per sample when applied in the main path of
    residual blocks.

    @type x: Tensor
    @param x: Input tensor.
    @type drop_prob: float
    @param drop_prob: Probability of dropping a path. Defaults to 0.0.
    @type scale_by_keep: bool
    @param scale_by_keep: Whether to scale the output by the keep probability. Defaults
        to True.
    @return: Tensor with dropped paths based on the provided drop probability.
    """
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor