import logging
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn import (
    Conv2d,
)
from luxonis_train.nodes.blocks import ConvModule

from luxonis_train.nodes.base_node import BaseNode

from .blocks import LCNetV3Block, LearnableRepLayer, make_divisible

logger = logging.getLogger(__name__)


NET_CONFIG_det = {
    "blocks2":
    # k, in_c, out_c, s, use_se
    [[3, 16, 32, 1, False]],
    "blocks3": [[3, 32, 64, 2, False], [3, 64, 64, 1, False]],
    "blocks4": [[3, 64, 128, 2, False], [3, 128, 128, 1, False]],
    "blocks5": [
        [3, 128, 256, 2, False],
        [5, 256, 256, 1, False],
        [5, 256, 256, 1, False],
        [5, 256, 256, 1, False],
        [5, 256, 256, 1, False],
    ],
    "blocks6": [
        [5, 256, 512, 2, True],
        [5, 512, 512, 1, True],
        [5, 512, 512, 1, False],
        [5, 512, 512, 1, False],
    ],
}

NET_CONFIG_rec = {
    "blocks2":
    # k, in_c, out_c, s, use_se
    [[3, 16, 32, 1, False]],
    "blocks3": [[3, 32, 64, 2, False], [3, 64, 64, 1, False]],
    "blocks4": [[3, 64, 128, 1, False], [3, 128, 128, 1, False]],
    "blocks5": [
        [3, 128, 256, 2, False],
        [5, 256, 256, 1, False],
        [5, 256, 256, 1, False],
        [5, 256, 256, 1, False],
        [5, 256, 256, 1, False],
    ],
    "blocks6": [
        [5, 256, 512, 1, True],
        [5, 512, 512, 1, True],
        [5, 512, 512, 1, False],
        [5, 512, 512, 1, False],
    ],
}


class PPLCNetV3(BaseNode[Tensor, list[Tensor]]):
    in_channels: int

    def __init__(
        self,
        scale: float = 0.95,
        conv_kxk_num: int = 4,
        det: bool = False,
        max_text_len: int = 40,
        **kwargs,
    ):  
        """PPLCNetV3 backbone.

        @see: U{Adapted from <https://github.com/PaddlePaddle/PaddleOCR/blob/main/ppocr/modeling/backbones/rec_lcnetv3.py>}
        @see: U{Original code <https://github.com/PaddlePaddle/PaddleOCR>}
        @license: U{Apache License, Version 2.0 <https://github.com/PaddlePaddle/PaddleOCR/blob/main/LICENSE>}
        @type scale: float
        @param scale: Scale factor. Defaults to 0.95.
        @type conv_kxk_num: int
        @param conv_kxk_num: Number of convolution branches. Defaults to 4.
        @type det: bool
        @param det: Whether to use the detection backbone. Defaults to False.
        @type max_text_len: int
        @param max_text_len: Maximum text length. Defaults to 40.
        """
        super().__init__(**kwargs)
        self.scale = scale
        self.det = det
        self.max_text_len = max_text_len

        self.net_config = NET_CONFIG_det if self.det else NET_CONFIG_rec

        self.conv1 = ConvModule(
            in_channels=self.in_channels,
            out_channels=make_divisible(16 * scale),
            kernel_size=3,
            stride=2,
            padding=1,
            bias=True,
            activation=nn.Identity(),
        )

        self.blocks2 = nn.Sequential(
            *[
                LCNetV3Block(
                    in_channels=make_divisible(in_c * scale),
                    out_channels=make_divisible(out_c * scale),
                    dw_size=k,
                    stride=s,
                    use_se=se,
                    conv_kxk_num=conv_kxk_num,
                )
                for _, (k, in_c, out_c, s, se) in enumerate(self.net_config["blocks2"])
            ]
        )

        self.blocks3 = nn.Sequential(
            *[
                LCNetV3Block(
                    in_channels=make_divisible(in_c * scale),
                    out_channels=make_divisible(out_c * scale),
                    dw_size=k,
                    stride=s,
                    use_se=se,
                    conv_kxk_num=conv_kxk_num,
                )
                for _, (k, in_c, out_c, s, se) in enumerate(self.net_config["blocks3"])
            ]
        )

        self.blocks4 = nn.Sequential(
            *[
                LCNetV3Block(
                    in_channels=make_divisible(in_c * scale),
                    out_channels=make_divisible(out_c * scale),
                    dw_size=k,
                    stride=s,
                    use_se=se,
                    conv_kxk_num=conv_kxk_num,
                )
                for _, (k, in_c, out_c, s, se) in enumerate(self.net_config["blocks4"])
            ]
        )

        self.blocks5 = nn.Sequential(
            *[
                LCNetV3Block(
                    in_channels=make_divisible(in_c * scale),
                    out_channels=make_divisible(out_c * scale),
                    dw_size=k,
                    stride=s,
                    use_se=se,
                    conv_kxk_num=conv_kxk_num,
                )
                for _, (k, in_c, out_c, s, se) in enumerate(self.net_config["blocks5"])
            ]
        )

        self.blocks6 = nn.Sequential(
            *[
                LCNetV3Block(
                    in_channels=make_divisible(in_c * scale),
                    out_channels=make_divisible(out_c * scale),
                    dw_size=k,
                    stride=s,
                    use_se=se,
                    conv_kxk_num=conv_kxk_num,
                )
                for _, (k, in_c, out_c, s, se) in enumerate(self.net_config["blocks6"])
            ]
        )
        self.out_channels = make_divisible(512 * scale)

        if self.det:
            mv_c = [16, 24, 56, 480]
            self.out_channels = [
                make_divisible(self.net_config["blocks3"][-1][2] * scale),
                make_divisible(self.net_config["blocks4"][-1][2] * scale),
                make_divisible(self.net_config["blocks5"][-1][2] * scale),
                make_divisible(self.net_config["blocks6"][-1][2] * scale),
            ]

            self.layer_list = nn.ModuleList(
                [
                    Conv2d(self.out_channels[0], int(mv_c[0] * scale), 1, 1, 0, bias=True),
                    Conv2d(self.out_channels[1], int(mv_c[1] * scale), 1, 1, 0, bias=True),
                    Conv2d(self.out_channels[2], int(mv_c[2] * scale), 1, 1, 0, bias=True),
                    Conv2d(self.out_channels[3], int(mv_c[3] * scale), 1, 1, 0, bias=True),
                ]
            )
            self.out_channels = [
                int(mv_c[0] * scale),
                int(mv_c[1] * scale),
                int(mv_c[2] * scale),
                int(mv_c[3] * scale),
            ]

    def set_export_mode(self, mode: bool = True) -> None:
        """Reparametrizes instances of L{LearnableRepLayer} in the network.

        @type mode: bool
        @param mode: Whether to set the export mode. Defaults to
            C{True}.
        """
        super().set_export_mode(mode)
        if self.export:
            logger.info("Reparametrizing 'LearnableRepLayer'.")
            for module in self.modules():
                if isinstance(module, LearnableRepLayer):
                    module.reparametrize()

    def forward(self, x: Tensor) -> list[Tensor] | Tensor:
        out_list = []
        x = self.conv1(x)

        x = self.blocks2(x)
        x = self.blocks3(x)
        out_list.append(x)
        x = self.blocks4(x)
        out_list.append(x)
        x = self.blocks5(x)
        out_list.append(x)
        x = self.blocks6(x)
        out_list.append(x)

        if self.det:
            out_list[0] = self.layer_list[0](out_list[0])
            out_list[1] = self.layer_list[1](out_list[1])
            out_list[2] = self.layer_list[2](out_list[2])
            out_list[3] = self.layer_list[3](out_list[3])
            return out_list

        x = F.adaptive_avg_pool2d(x, (1, self.max_text_len))

        return x