from luxonis_ml.typing import Params
from torch import Tensor, nn
from typing_extensions import override

from luxonis_train.nodes.heads import BaseHead
from luxonis_train.tasks import Tasks


class ClassificationHead(BaseHead):
    """Simple classification head.

    Consists of a global average pooling layer followed by a dropout
    layer and a single linear layer.

    Metadata:
        - Node type: head
        - Registry name: ``ClassificationHead``
        - Task: classification
        - Attach index: None
        - Inputs: ``features`` tensor
        - Outputs: classification logits tensor

    Provenance:
        - Source: Internal
        - License: Project license
        - Implementation notes: Applies global average pooling, dropout,
          and a linear classifier.

    Variants:
        - ``None``:
            - Default: yes
            - Aliases: None
            - Parameters:
                - No predefined variants.

    """

    in_channels: int
    task = Tasks.CLASSIFICATION
    parser: str = "ClassificationParser"

    def __init__(self, dropout_rate: float = 0.2, **kwargs):
        """Initialize the classification head.

        Args:
            dropout_rate (float): Dropout rate before last layer, range ``[0, 1]``. Defaults to ``0.2``.
            **kwargs (Any): Keyword arguments forwarded to the parent class.

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
        return {"is_softmax": False}
