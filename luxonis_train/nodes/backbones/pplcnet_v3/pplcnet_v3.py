from typing import Literal

from loguru import logger
from torch import Tensor, nn
from torch.nn import Conv2d
from torch.nn import functional as F

from luxonis_train.nodes.base_node import BaseNode
from luxonis_train.nodes.blocks import ConvModule

from .blocks import LCNetV3Block, LearnableRepLayer, make_divisible
from .variants import get_variant


class PPLCNetV3(BaseNode[Tensor, list[Tensor]]):
    in_channels: int

    def __init__(
        self,
        variant: Literal["rec-light"] = "rec-light",
        scale: float | None = None,
        conv_kxk_num: int | None = None,
        det: bool | None = None,
        net_config: dict[str, list[list[int | bool]]] | None = None,
        max_text_len: int = 40,
        **kwargs,
    ):
        """PPLCNetV3 backbone.

        @see: U{Adapted from <https://github.com/PaddlePaddle/PaddleOCR/
            blob/main/ppocr/modeling/backbones/rec_lcnetv3.py>}
        @see: U{Original code
            <https://github.com/PaddlePaddle/PaddleOCR>}
        @license: U{Apache License, Version 2.0
            <https://github.com/PaddlePaddle/PaddleOCR/blob/main/LICENSE
            >}
        @type scale: float
        @param scale: Scale factor. Defaults to 0.95.
        @type conv_kxk_num: int
        @param conv_kxk_num: Number of convolution branches. Defaults to
            4.
        @type det: bool
        @param det: Whether to use the detection backbone. Defaults to
            False.
        @type max_text_len: int
        @param max_text_len: Maximum text length. Defaults to 40.
        """
        super().__init__(**kwargs)

        var = get_variant(variant)

        self.scale = scale or var.scale
        self.det = det or var.det
        self.conv_kxk_num = conv_kxk_num or var.conv_kxk_num
        self.net_config = net_config or var.net_config

        self.max_text_len = max_text_len

        self.conv1 = ConvModule(
            in_channels=self.in_channels,
            out_channels=make_divisible(16 * self.scale),
            kernel_size=3,
            stride=2,
            padding=1,
            bias=True,
            activation=nn.Identity(),
        )

        self.blocks2 = nn.Sequential(
            *[
                LCNetV3Block(
                    in_channels=make_divisible(in_c * self.scale),
                    out_channels=make_divisible(out_c * self.scale),
                    dw_size=k,
                    stride=s,
                    use_se=se,  # type: ignore
                    conv_kxk_num=self.conv_kxk_num,
                )
                for _, (k, in_c, out_c, s, se) in enumerate(
                    self.net_config["blocks2"]
                )
            ]
        )

        self.blocks3 = nn.Sequential(
            *[
                LCNetV3Block(
                    in_channels=make_divisible(in_c * self.scale),
                    out_channels=make_divisible(out_c * self.scale),
                    dw_size=k,
                    stride=s,
                    use_se=se,  # type: ignore
                    conv_kxk_num=self.conv_kxk_num,
                )
                for _, (k, in_c, out_c, s, se) in enumerate(
                    self.net_config["blocks3"]
                )
            ]
        )

        self.blocks4 = nn.Sequential(
            *[
                LCNetV3Block(
                    in_channels=make_divisible(in_c * self.scale),
                    out_channels=make_divisible(out_c * self.scale),
                    dw_size=k,
                    stride=s,
                    use_se=se,  # type: ignore
                    conv_kxk_num=self.conv_kxk_num,
                )
                for _, (k, in_c, out_c, s, se) in enumerate(
                    self.net_config["blocks4"]
                )
            ]
        )

        self.blocks5 = nn.Sequential(
            *[
                LCNetV3Block(
                    in_channels=make_divisible(in_c * self.scale),
                    out_channels=make_divisible(out_c * self.scale),
                    dw_size=k,
                    stride=s,
                    use_se=se,  # type: ignore
                    conv_kxk_num=self.conv_kxk_num,
                )
                for _, (k, in_c, out_c, s, se) in enumerate(
                    self.net_config["blocks5"]
                )
            ]
        )

        self.blocks6 = nn.Sequential(
            *[
                LCNetV3Block(
                    in_channels=make_divisible(in_c * self.scale),
                    out_channels=make_divisible(out_c * self.scale),
                    dw_size=k,
                    stride=s,
                    use_se=se,  # type: ignore
                    conv_kxk_num=self.conv_kxk_num,
                )
                for _, (k, in_c, out_c, s, se) in enumerate(
                    self.net_config["blocks6"]
                )
            ]
        )
        self.out_channels = make_divisible(512 * self.scale)

        if self.det:
            mv_c = [16, 24, 56, 480]
            self.out_channels = [
                make_divisible(self.net_config["blocks3"][-1][2] * self.scale),
                make_divisible(self.net_config["blocks4"][-1][2] * self.scale),
                make_divisible(self.net_config["blocks5"][-1][2] * self.scale),
                make_divisible(self.net_config["blocks6"][-1][2] * self.scale),
            ]

            self.layer_list = nn.ModuleList(
                [
                    Conv2d(
                        self.out_channels[0],
                        int(mv_c[0] * self.scale),
                        1,
                        1,
                        0,
                        bias=True,
                    ),
                    Conv2d(
                        self.out_channels[1],
                        int(mv_c[1] * self.scale),
                        1,
                        1,
                        0,
                        bias=True,
                    ),
                    Conv2d(
                        self.out_channels[2],
                        int(mv_c[2] * self.scale),
                        1,
                        1,
                        0,
                        bias=True,
                    ),
                    Conv2d(
                        self.out_channels[3],
                        int(mv_c[3] * self.scale),
                        1,
                        1,
                        0,
                        bias=True,
                    ),
                ]
            )
            self.out_channels = [
                int(mv_c[0] * self.scale),
                int(mv_c[1] * self.scale),
                int(mv_c[2] * self.scale),
                int(mv_c[3] * self.scale),
            ]

    def set_export_mode(self, mode: bool = True) -> None:
        """Reparametrizes instances of L{LearnableRepLayer} in the
        network.

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

    def forward(self, x: Tensor) -> list[Tensor]:
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

        return [x]
