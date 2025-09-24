from typing import Literal

import torch
from luxonis_ml.typing import Params
from torch import Tensor, nn
from typing_extensions import override

from luxonis_train.nodes.heads import BaseHead
from luxonis_train.tasks import Tasks
from luxonis_train.utils.general import infer_upscale_factor


class DDRNetSegmentationHead(BaseHead):
    in_height: int
    in_width: int
    in_channels: int

    task = Tasks.SEGMENTATION
    parser: str = "SegmentationParser"

    def __init__(
        self,
        inter_channels: int = 64,
        inter_mode: Literal[
            "nearest",
            "linear",
            "bilinear",
            "bicubic",
            "trilinear",
            "area",
            "pixel_shuffle",
        ] = "bilinear",
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
        if (
            inter_mode == "pixel_shuffle"
            and inter_channels % (scale_factor**2) != 0
        ):
            raise ValueError(
                "For `pixel_shuffle`, inter_channels must be a "
                "multiple of scale_factor^2."
            )

        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.conv1 = nn.Conv2d(
            self.in_channels,
            inter_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(inter_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            inter_channels,
            inter_channels
            if inter_mode == "pixel_shuffle"
            else self.n_classes,
            kernel_size=1,
            padding=0,
            bias=True,
        )
        self.upscale = (
            nn.PixelShuffle(scale_factor)
            if inter_mode == "pixel_shuffle"
            else nn.Upsample(scale_factor=scale_factor, mode=inter_mode)
        )

    @override
    def load_checkpoint(
        self, path: str | None = None, strict: bool = False
    ) -> None:
        return super().load_checkpoint(path, strict=strict)

    @override
    def get_weights_url(self) -> str:
        if self.in_channels == 128:
            variant = "slim"
        elif self.in_channels == 256:
            variant = ""
        else:
            raise NotImplementedError(
                f"No online weights available for '{self.name}' "
                "with the chosen parameters"
            )
        return f"{{github}}/ddrnet_head_23{variant}_coco.ckpt"

    def forward(self, inputs: Tensor) -> Tensor:
        x: Tensor = self.relu(self.bn1(inputs))
        x = self.conv1(x)
        x = self.relu(self.bn2(x))
        x = self.conv2(x)
        x = self.upscale(x)
        if self.export:
            x = x.argmax(dim=1) if self.n_classes > 1 else x > 0
            return x.to(dtype=torch.int32)
        return x

    @override
    def get_custom_head_config(self) -> Params:
        """Returns custom head configuration.

        @rtype: dict
        @return: Custom head configuration.
        """
        return {"is_softmax": False}
