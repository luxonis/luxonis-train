"""ResNet backbone.

Source: U{https://pytorch.org/vision/main/models/resnet.html}
@license: U{PyTorch<https://github.com/pytorch/pytorch/blob/master/LICENSE>}
"""
from typing import Literal

import torchvision
from torch import Tensor, nn

from .base_node import BaseNode


class ResNet(BaseNode[Tensor, list[Tensor]]):
    def __init__(
        self,
        variant: Literal["18", "34", "50", "101", "152"] = "18",
        channels_list: list[int] | None = None,
        download_weights: bool = False,
        **kwargs,
    ):
        """Implementation of the ResNetX backbone.

        TODO: add more info

        @type variant: Literal["18", "34", "50", "101", "152"]
        @param variant: ResNet variant. Defaults to "18".
        @type channels_list: list[int] | None
        @param channels_list: List of channels to return.
            If unset, defaults to [64, 128, 256, 512].

        @type download_weights: bool
        @param download_weights: If True download weights from imagenet.
            Defaults to False.
        """
        super().__init__(**kwargs)

        if variant not in RESNET_VARIANTS:
            raise ValueError(
                f"ResNet model variant should be in {list(RESNET_VARIANTS.keys())}"
            )

        self.backbone = RESNET_VARIANTS[variant](
            weights="DEFAULT" if download_weights else None
        )

        self.backbone.fc = nn.Identity()

        self.channels_list = channels_list or [64, 128, 256, 512]

    def forward(self, inputs: Tensor) -> list[Tensor]:
        outs = []
        x = self.backbone.conv1(inputs)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        outs.append(x)
        x = self.backbone.layer2(x)
        outs.append(x)
        x = self.backbone.layer3(x)
        outs.append(x)
        x = self.backbone.layer4(x)
        outs.append(x)

        return outs


RESNET_VARIANTS = {
    "18": torchvision.models.resnet18,
    "34": torchvision.models.resnet34,
    "50": torchvision.models.resnet50,
    "101": torchvision.models.resnet101,
    "152": torchvision.models.resnet152,
}
