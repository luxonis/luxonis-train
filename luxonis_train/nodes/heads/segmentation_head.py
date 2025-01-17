from typing import Any

from torch import Tensor, nn

from luxonis_train.enums import TaskType
from luxonis_train.nodes.blocks import UpBlock
from luxonis_train.nodes.heads import BaseHead
from luxonis_train.utils import infer_upscale_factor


class SegmentationHead(BaseHead[Tensor, Tensor]):
    in_height: int
    in_width: int
    in_channels: int

    tasks: list[TaskType] = [TaskType.SEGMENTATION]
    parser: str = "SegmentationParser"

    def __init__(self, **kwargs: Any):
        """Basic segmentation FCN head.

        Adapted from: U{https://github.com/pytorch/vision/blob/main/torchvision/models/segmentation/fcn.py}
        @license: U{BSD-3 <https://github.com/pytorch/vision/blob/main/LICENSE>}
        """
        super().__init__(**kwargs)
        h, w = self.original_in_shape[1:]
        n_up = infer_upscale_factor((self.in_height, self.in_width), (h, w))

        modules: list[nn.Module] = []
        in_channels = self.in_channels
        for _ in range(int(n_up)):
            modules.append(
                UpBlock(in_channels=in_channels, out_channels=in_channels // 2)
            )
            in_channels //= 2

        self.head = nn.Sequential(
            *modules,
            nn.Conv2d(in_channels, self.n_classes, kernel_size=1),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.head(inputs)

    def get_custom_head_config(self) -> dict:
        """Returns custom head configuration.

        @rtype: dict
        @return: Custom head configuration.
        """
        return {
            "is_softmax": False,
        }
