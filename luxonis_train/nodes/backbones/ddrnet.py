"""DDRNet backbone.

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
    )  # Supports tensors of different dimensions
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


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


class DDRBackBoneBase(nn.Module, ABC):
    """Base class defining functions that must be supported by DDRBackBones."""

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


class BasicDDRBackBone(DDRBackBoneBase):
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


class DDRNet(BaseNode[Tensor, list[Tensor]]):
    def __init__(
        self,
        use_aux_heads: bool = True,
        upscale_module: nn.Module = None,
        highres_planes: int = 64,
        spp_width: int = 128,
        ssp_inter_mode: str = "bilinear",
        segmentation_inter_mode: str = "bilinear",
        block: Type[nn.Module] = BasicResNetBlock,
        skip_block: Type[nn.Module] = BasicResNetBlock,
        layer5_block: Type[nn.Module] = Bottleneck,
        layer5_bottleneck_expansion: int = 2,
        spp_kernel_sizes: list[int] = None,
        spp_strides: list[int] = None,
        layer3_repeats: int = 1,
        planes: int = 32,
        layers: list[int] = None,
        input_channels: int = 3,
        **kwargs,
    ):
        """Initialize the DDRNet with specified parameters.

        @type use_aux_heads: bool
        @param use_aux_heads: Whether to use auxiliary heads. Defaults to True.
        @type upscale_module: nn.Module
        @param upscale_module: Module for upscaling (e.g., bilinear interpolation).
            Defaults to UpscaleOnline().
        @type highres_planes: int
        @param highres_planes: Number of channels in the high resolution net. Defaults
            to 64.
        @type spp_width: int
        @param spp_width: Width of the branches in the SPP block. Defaults to 128.
        @type ssp_inter_mode: str
        @param ssp_inter_mode: Interpolation mode for the SPP block. Defaults to
            "bilinear".
        @type segmentation_inter_mode: str
        @param segmentation_inter_mode: Interpolation mode for the segmentation head.
            Defaults to "bilinear".
        @type block: Type[nn.Module]
        @param block: Type of block to use in the backbone. Defaults to
            BasicResNetBlock.
        @type skip_block: Type[nn.Module]
        @param skip_block: Type of block for skip connections. Defaults to
            BasicResNetBlock.
        @type layer5_block: Type[nn.Module]
        @param layer5_block: Type of block for layer5 and layer5_skip. Defaults to
            Bottleneck.
        @type layer5_bottleneck_expansion: int
        @param layer5_bottleneck_expansion: Expansion factor for Bottleneck block in
            layer5. Defaults to 2.
        @type spp_kernel_sizes: list[int]
        @param spp_kernel_sizes: Kernel sizes for the SPP module pooling. Defaults to
            [1, 5, 9, 17, 0].
        @type spp_strides: list[int]
        @param spp_strides: Strides for the SPP module pooling. Defaults to [1, 2, 4, 8,
            0].
        @type layer3_repeats: int
        @param layer3_repeats: Number of times to repeat the 3rd stage. Defaults to 1.
        @type planes: int
        @param planes: Base number of channels. Defaults to 32.
        @type layers: list[int]
        @param layers: Number of blocks in each layer of the backbone. Defaults to [2,
            2, 2, 2, 1, 2, 2, 1].
        @type input_channels: int
        @param input_channels: Number of input channels. Defaults to 3.
        @type kwargs: Any
        @param kwargs: Additional arguments to pass to L{BaseNode}.
        """

        if upscale_module is None:
            upscale_module = UpscaleOnline()
        if spp_kernel_sizes is None:
            spp_kernel_sizes = [1, 5, 9, 17, 0]
        if spp_strides is None:
            spp_strides = [1, 2, 4, 8, 0]
        if layers is None:
            layers = [2, 2, 2, 2, 1, 2, 2, 1]

        super().__init__(**kwargs)

        self._use_aux_heads = use_aux_heads
        self.upscale = upscale_module
        self.ssp_inter_mode = ssp_inter_mode
        self.segmentation_inter_mode = segmentation_inter_mode
        self.block = block
        self.skip_block = skip_block
        self.relu = nn.ReLU(inplace=False)
        self.layer3_repeats = layer3_repeats
        self.planes = planes
        self.layers = layers
        self.backbone_layers, self.additional_layers = self.layers[:4], self.layers[4:]
        self.input_channels = input_channels

        self._backbone: DDRBackBoneBase = BasicDDRBackBone(
            block=self.block,
            width=self.planes,
            layers=self.backbone_layers,
            input_channels=self.input_channels,
            layer3_repeats=self.layer3_repeats,
        )
        self._backbone.validate_backbone_attributes()
        out_chan_backbone = self._backbone.get_backbone_output_number_of_channels()

        # Define layers for layer 3
        self.compression3 = nn.ModuleList()
        self.down3 = nn.ModuleList()
        self.layer3_skip = nn.ModuleList()
        for i in range(layer3_repeats):
            self.compression3.append(
                ConvModule(
                    in_channels=out_chan_backbone["layer3"],
                    out_channels=highres_planes,
                    kernel_size=1,
                    bias=False,
                    activation=nn.Identity(),
                )
            )
            self.down3.append(
                ConvModule(
                    in_channels=highres_planes,
                    out_channels=out_chan_backbone["layer3"],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                    activation=nn.Identity(),
                )
            )
            self.layer3_skip.append(
                _make_layer(
                    in_planes=out_chan_backbone["layer2"] if i == 0 else highres_planes,
                    planes=highres_planes,
                    block=skip_block,
                    num_blocks=self.additional_layers[1],
                )
            )

        self.compression4 = ConvModule(
            in_channels=out_chan_backbone["layer4"],
            out_channels=highres_planes,
            kernel_size=1,
            bias=False,
            activation=nn.Identity(),
        )

        self.down4 = nn.Sequential(
            ConvModule(
                in_channels=highres_planes,
                out_channels=highres_planes * 2,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
                activation=nn.ReLU(inplace=True),
            ),
            ConvModule(
                in_channels=highres_planes * 2,
                out_channels=out_chan_backbone["layer4"],
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
                activation=nn.Identity(),
            ),
        )

        self.layer4_skip = _make_layer(
            block=skip_block,
            in_planes=highres_planes,
            planes=highres_planes,
            num_blocks=self.additional_layers[2],
        )
        self.layer5_skip = _make_layer(
            block=layer5_block,
            in_planes=highres_planes,
            planes=highres_planes,
            num_blocks=self.additional_layers[3],
            expansion=layer5_bottleneck_expansion,
        )

        self.layer5 = _make_layer(
            block=layer5_block,
            in_planes=out_chan_backbone["layer4"],
            planes=out_chan_backbone["layer4"],
            num_blocks=self.additional_layers[0],
            stride=2,
            expansion=layer5_bottleneck_expansion,
        )

        self.spp = DAPPM(
            in_planes=out_chan_backbone["layer4"] * layer5_bottleneck_expansion,
            branch_planes=spp_width,
            out_planes=highres_planes * layer5_bottleneck_expansion,
            inter_mode=self.ssp_inter_mode,
            kernel_sizes=spp_kernel_sizes,
            strides=spp_strides,
        )

        self.highres_planes = highres_planes
        self.layer5_bottleneck_expansion = layer5_bottleneck_expansion
        self.init_params()

    @property
    def backbone(self):
        """Create a fake backbone module to load backbone pre-trained weights."""
        return nn.Sequential(
            Dict(
                [
                    ("_backbone", self._backbone),
                    ("compression3", self.compression3),
                    ("compression4", self.compression4),
                    ("down3", self.down3),
                    ("down4", self.down4),
                    ("layer3_skip", self.layer3_skip),
                    ("layer4_skip", self.layer4_skip),
                    ("layer5_skip", self.layer5_skip),
                ]
            )
        )

    def forward(self, x: Tensor) -> list[Tensor]:
        width_output = x.shape[-1] // 8
        height_output = x.shape[-2] // 8

        x = self._backbone.stem(x)
        x = self._backbone.layer1(x)
        x = self._backbone.layer2(self.relu(x))

        # Repeat layer 3
        x_skip = x
        for i in range(self.layer3_repeats):
            out_layer3 = self._backbone.layer3[i](self.relu(x))
            out_layer3_skip = self.layer3_skip[i](self.relu(x_skip))

            x = out_layer3 + self.down3[i](self.relu(out_layer3_skip))
            x_skip = out_layer3_skip + self.upscale(
                self.compression3[i](self.relu(out_layer3)), height_output, width_output
            )

        # Save for auxiliary head
        if self._use_aux_heads:
            x_extra = x_skip

        out_layer4 = self._backbone.layer4(self.relu(x))
        out_layer4_skip = self.layer4_skip(self.relu(x_skip))

        x = out_layer4 + self.down4(self.relu(out_layer4_skip))
        x_skip = out_layer4_skip + self.upscale(
            self.compression4(self.relu(out_layer4)), height_output, width_output
        )

        out_layer5_skip = self.layer5_skip(self.relu(x_skip))

        x = self.upscale(
            self.spp(self.layer5(self.relu(x))), height_output, width_output
        )

        x = x + out_layer5_skip

        if self._use_aux_heads:
            return [x, x_extra]
        else:
            return [x]

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
