import torch
from torch import nn
from torch.nn import functional as F

from luxonis_train.nodes.backbones.efficientvit.blocks import (
    DepthwiseSeparableConv,
)
from luxonis_train.nodes.blocks import ConvModule


class UAFM(nn.Module):
    """The base of Unified Attention Fusion Module.

    Args:
        x_ch (int): The channel of x tensor, which is the low level feature.
        y_ch (int): The channel of y tensor, which is the high level feature.
        out_ch (int): The channel of output tensor.
        ksize (int, optional): The kernel size of the conv for x tensor. Default: 3.
        resize_mode (str, optional): The resize model in unsampling y tensor. Default: bilinear.
    """

    def __init__(self, x_ch, y_ch, out_ch, ksize=3, resize_mode="bilinear"):
        super().__init__()

        self.conv_x = ConvModule(
            in_channels=x_ch,
            out_channels=y_ch,
            kernel_size=ksize,
            padding=ksize // 2,
            bias=False,
        )
        self.conv_out = ConvModule(
            in_channels=y_ch,
            out_channels=out_ch,
            kernel_size=3,
            padding=1,
            bias=False,
        )

        self.resize_mode = resize_mode

    def check(self, x, y):
        assert x.ndim == 4 and y.ndim == 4
        x_h, x_w = x.shape[2:]
        y_h, y_w = y.shape[2:]
        assert x_h >= y_h and x_w >= y_w

    def prepare(self, x, y):
        x = self.prepare_x(x, y)
        y = self.prepare_y(x, y)
        return x, y

    def prepare_x(self, x, y):
        x = self.conv_x(x)
        return x

    def prepare_y(self, x, y):
        y_up = F.interpolate(y, x.shape[2:], mode=self.resize_mode)
        return y_up

    def fuse(self, x, y):
        out = x + y
        out = self.conv_out(out)
        return out

    def forward(self, x, y):
        """
        Args:
            x (Tensor): The low level feature.
            y (Tensor): The high level feature.
        """
        self.check(x, y)
        x, y = self.prepare(x, y)
        out = self.fuse(x, y)
        return out


class UAFMMobile(UAFM):
    """Unified Attention Fusion Module for mobile.

    Args:
        x_ch (int): The channel of x tensor, which is the low level feature.
        y_ch (int): The channel of y tensor, which is the high level feature.
        out_ch (int): The channel of output tensor.
        ksize (int, optional): The kernel size of the conv for x tensor. Default: 3.
        resize_mode (str, optional): The resize model in unsampling y tensor. Default: bilinear.
    """

    def __init__(self, x_ch, y_ch, out_ch, ksize=3, resize_mode="bilinear"):
        super().__init__(x_ch, y_ch, out_ch, ksize, resize_mode)

        self.conv_x = DepthwiseSeparableConv(
            in_channels=x_ch,
            out_channels=y_ch,
            kernel_size=ksize,
            stride=1,
            padding=ksize // 2,
            use_bias=[False, False],
            activation=[nn.Identity(), nn.ReLU()],
            use_residual=False,
        )
        self.conv_out = DepthwiseSeparableConv(
            in_channels=y_ch,
            out_channels=out_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            use_bias=[False, False],
            activation=[nn.Identity(), nn.ReLU()],
            use_residual=False,
        )


class PPSegHead(nn.Module):
    """The head of MobileSeg.

    Args:
        backbone_out_chs (List(Tensor)): The channels of output tensors in the backbone.
        arm_out_chs (List(int)): The out channels of each arm module.
        cm_bin_sizes (List(int)): The bin size of context module.
        cm_out_ch (int): The output channel of the last context module.
        arm_type (str): The type of attention refinement module.
        resize_mode (str): The resize mode for the upsampling operation in decoder.
    """

    def __init__(
        self,
        backbone_out_chs,
        arm_out_chs,
        cm_bin_sizes,
        cm_out_ch,
        arm_type,
        resize_mode,
        use_last_fuse,
    ):
        super().__init__()

        self.cm = MobileContextModule(
            backbone_out_chs[-1], cm_out_ch, cm_out_ch, cm_bin_sizes
        )

        assert arm_type == "UAFMMobile", "Not support arm_type ({})".format(
            arm_type
        )
        arm_class: UAFMMobile = eval("UAFMMobile")

        self.arm_list = nn.ModuleList()  # [..., arm8, arm16, arm32]
        for i in range(len(backbone_out_chs)):
            low_chs = backbone_out_chs[i]
            high_ch = (
                cm_out_ch
                if i == len(backbone_out_chs) - 1
                else arm_out_chs[i + 1]
            )
            out_ch = arm_out_chs[i]
            arm = arm_class(
                low_chs, high_ch, out_ch, ksize=3, resize_mode=resize_mode
            )
            self.arm_list.append(arm)

        self.use_last_fuse = use_last_fuse
        if self.use_last_fuse:
            self.fuse_convs = nn.ModuleList()
            for i in range(1, len(arm_out_chs)):
                conv = DepthwiseSeparableConv(
                    in_channels=arm_out_chs[i],
                    out_channels=arm_out_chs[0],
                    kernel_size=3,
                    padding="same",
                    use_bias=[False, False],
                    activation=[nn.Identity(), nn.ReLU()],
                    use_residual=False,
                )
                self.fuse_convs.append(conv)
            self.last_conv = DepthwiseSeparableConv(
                in_channels=len(arm_out_chs) * arm_out_chs[0],
                out_channels=arm_out_chs[0],
                kernel_size=3,
                padding="same",
                use_bias=[False, False],
                activation=[nn.Identity(), nn.ReLU()],
                use_residual=False,
            )

    def forward(self, in_feat_list):
        """
        Args:
            in_feat_list (List(Tensor)): Such as [x2, x4, x8, x16, x32].
                x2, x4 and x8 are optional.
        Returns:
            out_feat_list (List(Tensor)): Such as [x2, x4, x8, x16, x32].
                x2, x4 and x8 are optional.
                The length of in_feat_list and out_feat_list are the same.
        """

        high_feat = self.cm(in_feat_list[-1])
        out_feat_list = []

        for i in reversed(range(len(in_feat_list))):
            low_feat = in_feat_list[i]
            arm = self.arm_list[i]
            high_feat = arm(low_feat, high_feat)
            out_feat_list.insert(0, high_feat)

        if self.use_last_fuse:
            x_list = [out_feat_list[0]]
            size = out_feat_list[0].shape[2:]
            for x, conv in zip(out_feat_list[1:], self.fuse_convs):
                x = conv(x)
                x = F.interpolate(
                    x, size=size, mode="bilinear", align_corners=False
                )
                x_list.append(x)
            x = torch.cat(x_list, dim=1)
            x = self.last_conv(x)
            out_feat_list[0] = x

        return out_feat_list


class MobileContextModule(nn.Module):
    """Context Module for Mobile Model.

    Args:
        in_channels (int): The number of input channels to pyramid pooling module.
        inter_channels (int): The number of inter channels to pyramid pooling module.
        out_channels (int): The number of output channels after pyramid pooling module.
        bin_sizes (tuple, optional): The out size of pooled feature maps. Default: (1, 3).
        align_corners (bool): An argument of F.interpolate. It should be set to False
            when the output size of feature is even, e.g. 1024x512, otherwise it is True, e.g. 769x769.
    """

    def __init__(
        self,
        in_channels,
        inter_channels,
        out_channels,
        bin_sizes,
        align_corners=False,
    ):
        super().__init__()

        self.stages = nn.ModuleList(
            [
                self._make_stage(in_channels, inter_channels, size)
                for size in bin_sizes
            ]
        )

        self.conv_out = DepthwiseSeparableConv(
            in_channels=inter_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding="same",
            use_bias=[False, False],
            activation=[nn.Identity(), nn.ReLU()],
            use_residual=False,
        )

        self.align_corners = align_corners

    def _make_stage(self, in_channels, out_channels, size):
        prior = nn.AdaptiveAvgPool2d(output_size=size)
        conv = ConvModule(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            padding="same",
        )
        return nn.Sequential(prior, conv)

    def forward(self, input):
        out = None
        input_shape = input.shape[2:]

        for stage in self.stages:
            x = stage(input)
            x = F.interpolate(
                x,
                input_shape,
                mode="bilinear",
                align_corners=self.align_corners,
            )
            if out is None:
                out = x
            else:
                out += x

        out = self.conv_out(out)
        return out


class SegHead(nn.Module):
    def __init__(self, in_chan, mid_chan, n_classes):
        super().__init__()
        self.conv = DepthwiseSeparableConv(
            in_channels=in_chan,
            out_channels=mid_chan,
            kernel_size=3,
            padding="same",
            use_bias=[False, False],
            activation=[nn.Identity(), nn.ReLU()],
            use_residual=False,
        )
        self.conv_out = nn.Conv2d(
            mid_chan, n_classes, kernel_size=1, bias=False
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        return x
