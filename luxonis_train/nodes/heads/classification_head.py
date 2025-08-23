from luxonis_ml.typing import Params
from torch import Tensor, nn
from typing_extensions import override

from luxonis_train.nodes.heads import BaseHead
from luxonis_train.tasks import Tasks


class ClassificationHead(BaseHead):
    in_channels: int
    task = Tasks.CLASSIFICATION
    parser: str = "ClassificationParser"

    def __init__(self, dropout_rate: float = 0.2, **kwargs):
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

    @override
    def get_custom_head_config(self) -> Params:
        """Returns custom head configuration.

        @rtype: dict
        @return: Custom head configuration.
        """
        return {"is_softmax": False}
