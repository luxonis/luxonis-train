from luxonis_ml.typing import Params
from torch import Tensor, nn
from typing_extensions import override

from luxonis_train.nodes.heads import BaseHead
from luxonis_train.tasks import Tasks


class TransformerClassificationHead(BaseHead):
    """Classification decoder head for CLS token output from DINOv3.

    Converts [B, C] (CLS token embedding) to [B, n_classes].
    """

    attach_index = -1
    task = Tasks.CLASSIFICATION
    parser: str = "ClassificationParser"

    def __init__(self, dropout_rate: float = 0.2, **kwargs):
        """Classification head for transformer CLS tokens.

        @param dropout_rate: Dropout rate before last layer.
        """
        super().__init__(**kwargs)

        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(self.in_channels, self.n_classes)

    @property
    def in_channels(self) -> int:
        result = self._get_nth_size(-1)
        if isinstance(result, list):
            raise TypeError("Expected a single [B, C], got multiple.")
        return result

    def forward(self, x: Tensor) -> Tensor:
        """
        @param x: CLS tensor in the form [B, C], where C is the embedding dim.
        @type x: Tensor
        @return: Class logits [B, n_classes]

        @note: Steps performed:
            1) Apply dropout to the CLS token.
            2) Apply a linear layer to produce class logits.
        """
        x = self.dropout(x)
        return self.fc(x)

    @override
    def get_custom_head_config(self) -> Params:
        """Returns custom head configuration.

        @rtype: dict
        @return: Custom head configuration.
        """
        return {"is_softmax": False}
