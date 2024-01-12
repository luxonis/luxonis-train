from torch import Tensor, nn

from luxonis_train.utils.types import LabelType, Packet

from .base_node import BaseNode


class ClassificationHead(BaseNode[Tensor, Tensor]):
    in_channels: int
    attach_index: int = -1

    def __init__(
        self,
        dropout_rate: float = 0.2,
        **kwargs,
    ):
        """Simple classification head.

        @type dropout_rate: float
        @param dropout_rate: Dropout rate before last layer, range C{[0, 1]}. Defaults
            to C{0.2}.
        """
        super().__init__(task_type=LabelType.CLASSIFICATION, **kwargs)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.in_channels, self.n_classes),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.head(inputs)

    def wrap(self, output: Tensor) -> Packet[Tensor]:
        return {"classes": [output]}
