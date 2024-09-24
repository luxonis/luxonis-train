from typing import Any

from torch import Tensor, nn

from luxonis_train.enums import TaskType
from luxonis_train.nodes.base_node import BaseNode


class ClassificationHead(BaseNode[Tensor, Tensor]):
    in_channels: int
    tasks: list[TaskType] = [TaskType.CLASSIFICATION]

    def __init__(self, dropout_rate: float = 0.2, **kwargs: Any):
        """Simple classification head.

        Consists of a global average pooling layer followed by a dropout
        layer and a single linear layer.

        @type dropout_rate: float
        @param dropout_rate: Dropout rate before last layer, range C{[0,
            1]}. Defaults to C{0.2}.
        """
        super().__init__(**kwargs)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.in_channels, self.n_classes),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.head(inputs)
