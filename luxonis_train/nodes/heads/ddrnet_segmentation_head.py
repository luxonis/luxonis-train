import logging

import torch
import torch.nn as nn
from torch import Tensor

from luxonis_train.nodes.base_node import BaseNode
from luxonis_train.nodes.blocks import ConvModule
from luxonis_train.utils.general import infer_upscale_factor
from luxonis_train.utils.types import LabelType

logger = logging.getLogger(__name__)


class DDRNetSegmentationHead(BaseNode[Tensor, Tensor]):
    attach_index: int = -1

    tasks: list[LabelType] = [LabelType.SEGMENTATION]

    def __init__(
        self,
        inter_channels: int = 64,
        inter_mode: str = "bilinear",
        **kwargs,
    ):
        """DDRNet segmentation head.

        @see: U{Adapted from <https://github.com/Deci-AI/super-gradients/blob/master/src
            /super_gradients/training/models/segmentation_models/ddrnet.py>}
        @see: U{Original code <https://github.com/ydhongHIT/DDRNet>}
        @see: U{Paper <https://arxiv.org/pdf/2101.06085.pdf>}
        @license: U{Apache License, Version 2.0 <https://github.com/Deci-AI/super-
            gradients/blob/master/LICENSE.md>}
        @type inter_channels: int
        @param inter_channels: Width of internal conv. Must be a multiple of
            scale_factor^2 when inter_mode is pixel_shuffle. Defaults to 64.
        @type inter_mode: str
        @param inter_mode: Upsampling method. One of nearest, linear, bilinear, bicubic,
            trilinear, area or pixel_shuffle. If pixel_shuffle is set, nn.PixelShuffle
            is used for scaling. Defaults to "bilinear".
        """
        super().__init__(**kwargs)
        model_in_h, model_in_w = self.original_in_shape[1:]
        scale_factor = 2 ** infer_upscale_factor(
            (self.in_height, self.in_width), (model_in_h, model_in_w)
        )
        self.scale_factor = scale_factor

        if inter_mode == "pixel_shuffle":
            if inter_channels % (scale_factor**2) != 0:
                raise ValueError(
                    "When using pixel_shuffle, inter_channels must be a multiple of scale_factor^2."
                )

        self.conv1 = ConvModule(
            self.in_channels,
            inter_channels,
            kernel_size=3,
            padding=1,
            bias=False,
            activation=nn.ReLU(inplace=True),
        )

        if inter_mode == "pixel_shuffle":
            self.conv2 = ConvModule(
                inter_channels,
                inter_channels,
                kernel_size=1,
                padding=0,
                bias=True,
                activation=nn.Identity(),
            )
            self.upscale = nn.PixelShuffle(scale_factor)
        else:
            self.conv2 = ConvModule(
                inter_channels,
                self.n_classes,
                kernel_size=1,
                padding=0,
                bias=True,
                activation=nn.Identity(),
            )
            self.upscale = nn.Upsample(scale_factor=scale_factor, mode=inter_mode)

    def forward(self, inputs: Tensor) -> Tensor:
        x = self.conv1(inputs)
        out = self.conv2(x)
        out = self.upscale(out)

        return out

    def set_export_mode(self, mode: bool = True) -> None:
        """Sets the module to export mode.

        Replaces the forward method with an identity function when in export mode.

        @warning: The replacement is destructive and cannot be undone.
        @type mode: bool
        @param mode: Whether to set the export mode to True or False. Defaults to True.
        """
        super().set_export_mode(mode)
        if self.export and self.attach_index != -1:
            logger.info("Removing the auxiliary head.")

            self.forward = lambda x: torch.tensor([])
