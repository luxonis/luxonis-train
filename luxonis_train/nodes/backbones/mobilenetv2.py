"""MobileNetV2 backbone.

TODO: source?
"""

import torchvision
from torch import Tensor, nn

from luxonis_train.nodes.base_node import BaseNode


class MobileNetV2(BaseNode[Tensor, list[Tensor]]):
    """Implementation of the MobileNetV2 backbone.

    TODO: add more info
    """

    def __init__(self, download_weights: bool = False, **kwargs):
        """Constructor of the MobileNetV2 backbone.

        @type download_weights: bool
        @param download_weights: If True download weights from imagenet. Defaults to
            False.
        @type kwargs: Any
        @param kwargs: Additional arguments to pass to L{BaseNode}.
        """
        super().__init__(**kwargs)

        mobilenet_v2 = torchvision.models.mobilenet_v2(
            weights="DEFAULT" if download_weights else None
        )
        mobilenet_v2.classifier = nn.Identity()
        self.out_indices = [3, 6, 13, 18]
        self.channels = [24, 32, 96, 1280]
        self.backbone = mobilenet_v2

    def forward(self, x: Tensor) -> list[Tensor]:
        outs = []
        for i, module in enumerate(self.backbone.features):
            x = module(x)
            if i in self.out_indices:
                outs.append(x)

        return outs
