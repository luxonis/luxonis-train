"""DDRNet backbone.

Adapted from: U{https://github.com/Deci-AI/super-gradients/blob/master/src/super_gradients/training/models/segmentation_models/ddrnet.py}
Original source: U{https://github.com/ydhongHIT/DDRNet}
Paper: U{https://arxiv.org/pdf/2101.06085.pdf}
@license: U{https://github.com/Deci-AI/super-gradients/blob/master/LICENSE.md}
"""
from typing import Literal
from abc import ABC
from typing import Optional, Callable, Union, List, Tuple, Dict


import torchvision
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..base_node import BaseNode

def ConvBN(in_channels: int, out_channels: int, kernel_size: int, bias=True, stride=1, padding=0, add_relu=False):
    seq = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias, stride=stride, padding=padding), nn.BatchNorm2d(out_channels)]
    if add_relu:
        seq.append(nn.ReLU(inplace=True))
    return nn.Sequential(*seq)


def _make_layer(block, in_planes, planes, num_blocks, stride=1, expansion=1):
    layers = []
    layers.append(block(in_planes, planes, stride, final_relu=num_blocks > 1, expansion=expansion))
    in_planes = planes * expansion
    if num_blocks > 1:
        for i in range(1, num_blocks):
            if i == (num_blocks - 1):
                layers.append(block(in_planes, planes, stride=1, final_relu=False, expansion=expansion))
            else:
                layers.append(block(in_planes, planes, stride=1, final_relu=True, expansion=expansion))

    return nn.Sequential(*layers)

def drop_path(x, drop_prob: float = 0.0, scale_by_keep: bool = True):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    """

    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).

    Intended usage of this block is the following:

    >>> class ResNetBlock(nn.Module):
    >>>   def __init__(self, ..., drop_path_rate:float):
    >>>     self.drop_path = DropPath(drop_path_rate)
    >>>
    >>>   def forward(self, x):
    >>>     return x + self.drop_path(self.conv_bn_act(x))

    Code taken from TIMM (https://github.com/rwightman/pytorch-image-models)
    Apache License 2.0
    """

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        """

        :param drop_prob: Probability of zeroing out individual vector (channel dimension) of each feature map
        :param scale_by_keep: Whether to scale the output by the keep probability. Enable by default and helps to
                              keep output mean & std in the same range as w/o drop path.
        """
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x

        return drop_path(x, self.drop_prob, self.scale_by_keep)

    def extra_repr(self):
        return f"drop_prob={round(self.drop_prob,3):0.3f}"


class BasicResNetBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1, expansion=1, final_relu=True, droppath_prob=0.0):
        super(BasicResNetBlock, self).__init__()
        self.expansion = expansion
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.final_relu = final_relu

        self.drop_path = DropPath(drop_prob=droppath_prob)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.drop_path(out)
        out += self.shortcut(x)
        if self.final_relu:
            out = F.relu(out)
        return out

class Bottleneck(nn.Module):
    def __init__(self, in_planes, planes, stride=1, expansion=4, final_relu=True, droppath_prob=0.0):
        super(Bottleneck, self).__init__()
        self.expansion = expansion
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.final_relu = final_relu

        self.drop_path = DropPath(drop_prob=droppath_prob)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        out = self.drop_path(out)

        out += self.shortcut(x)

        if self.final_relu:
            out = F.relu(out)

        return out


class DAPPMBranch(nn.Module):
    def __init__(self, kernel_size: int, stride: int, in_planes: int, branch_planes: int, inter_mode: str = "bilinear"):
        """
        A DAPPM branch
        :param kernel_size: the kernel size for the average pooling
                when stride=0 this parameter is omitted and AdaptiveAvgPool2d over all the input is performed
        :param stride: stride for the average pooling
                when stride=0: an AdaptiveAvgPool2d over all the input is performed (output is 1x1)
                when stride=1: no average pooling is performed
                when stride>1: average polling is performed (scaling the input down and up again)
        :param in_planes:
        :param branch_planes: width after the the first convolution
        :param inter_mode: interpolation mode for upscaling
        """

        super().__init__()
        down_list = []
        if stride == 0:
            # when stride is 0 average pool all the input to 1x1
            down_list.append(nn.AdaptiveAvgPool2d((1, 1)))
        elif stride == 1:
            # when stride id 1 no average pooling is used
            pass
        else:
            down_list.append(nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=stride))

        down_list.append(nn.BatchNorm2d(in_planes))
        down_list.append(nn.ReLU(inplace=True))
        down_list.append(nn.Conv2d(in_planes, branch_planes, kernel_size=1, bias=False))

        self.down_scale = nn.Sequential(*down_list)
        self.up_scale = UpscaleOnline(inter_mode)

        if stride != 1:
            self.process = nn.Sequential(
                nn.BatchNorm2d(branch_planes),
                nn.ReLU(inplace=True),
                nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
            )

    def forward(self, x):
        """
        All branches of the DAPPM but the first one receive the output of the previous branch as a second input
        :param x: in branch 0 - the original input of the DAPPM. in other branches - a list containing the original
        input and the output of the previous branch.
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
    def __init__(self, in_planes: int, branch_planes: int, out_planes: int, kernel_sizes: list, strides: list, inter_mode: str = "bilinear"):
        super().__init__()

        assert len(kernel_sizes) == len(strides), "len of kernel_sizes and strides must be the same"
        self.branches = nn.ModuleList()
        for kernel_size, stride in zip(kernel_sizes, strides):
            self.branches.append(DAPPMBranch(kernel_size=kernel_size, stride=stride, in_planes=in_planes, branch_planes=branch_planes, inter_mode=inter_mode))

        self.compression = nn.Sequential(
            nn.BatchNorm2d(branch_planes * len(self.branches)),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes * len(self.branches), out_planes, kernel_size=1, bias=False),
        )
        self.shortcut = nn.Sequential(
            nn.BatchNorm2d(in_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False),
        )

    def forward(self, x):
        x_list = []
        for i, branch in enumerate(self.branches):
            if i == 0:
                x_list.append(branch(x))
            else:
                x_list.append(branch([x, x_list[i - 1]]))

        out = self.compression(torch.cat(x_list, 1)) + self.shortcut(x)
        return out


class UpscaleOnline(nn.Module):
    """
    In some cases the required scale/size for the scaling is known only when the input is received.
    This class support such cases. only the interpolation mode is set in advance.
    """

    def __init__(self, mode="bilinear"):
        super().__init__()
        self.mode = mode

    def forward(self, x, output_height: int, output_width: int):
        return F.interpolate(x, size=[output_height, output_width], mode=self.mode)


class DDRBackBoneBase(nn.Module, ABC):
    """A base class defining functions that must be supported by DDRBackBones"""

    def validate_backbone_attributes(self):
        expected_attributes = ["stem", "layer1", "layer2", "layer3", "layer4", "input_channels"]
        for attribute in expected_attributes:
            assert hasattr(self, attribute), f"Invalid backbone - attribute '{attribute}' is missing"

    def get_backbone_output_number_of_channels(self):
        """Return a dictionary of the shapes of each output of the backbone to determine the in_channels of the
        skip and compress layers"""
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
    def __init__(self, block: nn.Module.__class__, width: int, layers: list, input_channels: int, layer3_repeats: int = 1):
        super().__init__()
        self.input_channels = input_channels
        self.stem = nn.Sequential(
            ConvBN(in_channels=input_channels, out_channels=width, kernel_size=3, stride=2, padding=1, add_relu=True),
            ConvBN(in_channels=width, out_channels=width, kernel_size=3, stride=2, padding=1, add_relu=True),
        )
        self.layer1 = _make_layer(block=block, in_planes=width, planes=width, num_blocks=layers[0])
        self.layer2 = _make_layer(block=block, in_planes=width, planes=width * 2, num_blocks=layers[1], stride=2)
        self.layer3 = nn.ModuleList(
            [_make_layer(block=block, in_planes=width * 2, planes=width * 4, num_blocks=layers[2], stride=2)]
            + [_make_layer(block=block, in_planes=width * 4, planes=width * 4, num_blocks=layers[2], stride=1) for _ in range(layer3_repeats - 1)]
        )
        self.layer4 = _make_layer(block=block, in_planes=width * 4, planes=width * 8, num_blocks=layers[3], stride=2)

    def replace_input_channels(self, in_channels: int, compute_new_weights_fn: Optional[Callable[[nn.Module, int], nn.Module]] = None):
        from super_gradients.modules.weight_replacement_utils import replace_conv2d_input_channels

        self.stem[0][0] = replace_conv2d_input_channels(conv=self.stem[0][0], in_channels=in_channels, fn=compute_new_weights_fn)
        self.input_channels = self.get_input_channels()

    def get_input_channels(self) -> int:
        return self.stem[0][0].in_channels


class DDRNet(BaseNode[Tensor, list[Tensor]]):
    def __init__(
        self,
        #backbone: DDRBackBoneBase.__class__,
        use_aux_heads: bool = True,
        upscale_module: nn.Module = UpscaleOnline(),
        highres_planes: int = 64,
        spp_width: int = 128,
        #head_width: int,
        ssp_inter_mode: str = "bilinear",
        segmentation_inter_mode: str = "bilinear",
        block: nn.Module.__class__ = BasicResNetBlock,
        skip_block: nn.Module.__class__ = BasicResNetBlock,
        layer5_block: nn.Module.__class__ = Bottleneck,
        layer5_bottleneck_expansion: int = 2,
        #classification_mode=False,
        spp_kernel_sizes: list = [1, 5, 9, 17, 0],
        spp_strides: list = [1, 2, 4, 8, 0],
        layer3_repeats: int = 1,
        planes: int = 32,
        layers: list = [2, 2, 2, 2, 1, 2, 2, 1],
        input_channels: int = 3,
        **kwargs,
    ):
        """

        :param upscale_module: upscale to use in the backbone (DAPPM and Segmentation head are using bilinear interpolation)
        :param highres_planes: number of channels in the high resolution net
        :param ssp_inter_mode: the interpolation used in the SPP block
        :param segmentation_inter_mode: the interpolation used in the segmentation head
        :param skip_block: allows specifying a different block (from 'block') for the skip layer
        :param layer5_block: type of block to use in layer5 and layer5_skip
        :param layer5_bottleneck_expansion: determines the expansion rate for Bottleneck block
        :param spp_kernel_sizes: list of kernel sizes for the spp module pooling
        :param spp_strides: list of strides for the spp module pooling
        :param layer3_repeats: number of times to repeat the 3rd stage of ddr model, including the paths interchange
         modules.
        """

        super().__init__(**kwargs)
        #self.use_aux_heads = use_aux_heads
        self._use_aux_heads = use_aux_heads
        self.upscale = upscale_module
        self.ssp_inter_mode = ssp_inter_mode
        self.segmentation_inter_mode = segmentation_inter_mode
        self.block = block
        self.skip_block = skip_block
        self.relu = nn.ReLU(inplace=False)
        #self.classification_mode = classification_mode
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

        # Repeat r-times layer4
        self.compression3, self.down3, self.layer3_skip = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for i in range(layer3_repeats):
            self.compression3.append(ConvBN(in_channels=out_chan_backbone["layer3"], out_channels=highres_planes, kernel_size=1, bias=False))
            self.down3.append(ConvBN(in_channels=highres_planes, out_channels=out_chan_backbone["layer3"], kernel_size=3, stride=2, padding=1, bias=False))
            self.layer3_skip.append(
                _make_layer(
                    in_planes=out_chan_backbone["layer2"] if i == 0 else highres_planes,
                    planes=highres_planes,
                    block=skip_block,
                    num_blocks=self.additional_layers[1],
                )
            )

        self.compression4 = ConvBN(in_channels=out_chan_backbone["layer4"], out_channels=highres_planes, kernel_size=1, bias=False)

        self.down4 = nn.Sequential(
            ConvBN(in_channels=highres_planes, out_channels=highres_planes * 2, kernel_size=3, stride=2, padding=1, bias=False, add_relu=True),
            ConvBN(in_channels=highres_planes * 2, out_channels=out_chan_backbone["layer4"], kernel_size=3, stride=2, padding=1, bias=False),
        )
        self.layer4_skip = _make_layer(block=skip_block, in_planes=highres_planes, planes=highres_planes, num_blocks=self.additional_layers[2])
        self.layer5_skip = _make_layer(
            block=layer5_block, in_planes=highres_planes, planes=highres_planes, num_blocks=self.additional_layers[3], expansion=layer5_bottleneck_expansion
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
        #self.head_width = head_width
        self.init_params()

    @property
    def backbone(self):
        """
        Create a fake backbone module to load backbone pre-trained weights.
        """
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
                    ("layer4_skip", self.layer4_skip),
                    ("layer5_skip", self.layer5_skip),
                ]
            )
        )

    def forward(self, x: Tensor) -> List[Tensor]:
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
            x_skip = out_layer3_skip + self.upscale(self.compression3[i](self.relu(out_layer3)), height_output, width_output)

        # save for auxiliary head
        if self._use_aux_heads:
            x_extra = x_skip

        out_layer4 = self._backbone.layer4(self.relu(x))
        out_layer4_skip = self.layer4_skip(self.relu(x_skip))

        x = out_layer4 + self.down4(self.relu(out_layer4_skip))
        x_skip = out_layer4_skip + self.upscale(self.compression4(self.relu(out_layer4)), height_output, width_output)

        out_layer5_skip = self.layer5_skip(self.relu(x_skip))

        # if self.classification_mode:
        #     x_skip = self.high_to_low_fusion(self.relu(out_layer5_skip))
        #     x = self.layer5(self.relu(x))
        #     x = self.average_pool(x + x_skip)
        #     x = self.fc(x.squeeze())
        #     return x
        # else:
        x = self.upscale(self.spp(self.layer5(self.relu(x))), height_output, width_output)

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

    @property
    def use_aux_heads(self):
        return self._use_aux_heads

    @use_aux_heads.setter
    def use_aux_heads(self, use_aux: bool):
        """
        public setter for self._use_aux_heads, called every time an assignment to self.use_aux_heads is applied.
        if use_aux is False, `_remove_auxiliary_heads` is called to delete auxiliary and detail heads.
        if use_aux is True, and self._use_aux_heads was already set to False a ValueError is raised, recreating
            aux and detail heads outside init method is not allowed, and the module should be recreated.
        """
        if use_aux is True and self._use_aux_heads is False:
            raise ValueError(
                "Cant turn use_aux_heads from False to True. Try initiating the module again with"
                " `use_aux_heads=True` or initiating the auxiliary heads modules manually."
            )
        if not use_aux:
            self._remove_auxiliary_heads()
        self._use_aux_heads = use_aux

    def prep_model_for_conversion(self, input_size: Union[tuple, list] = None, **kwargs):
        # set to false and delete auxiliary and detail heads modules.
        self.use_aux_heads = False

    def _remove_auxiliary_heads(self):
        if hasattr(self, "seghead_extra"):
            del self.seghead_extra
