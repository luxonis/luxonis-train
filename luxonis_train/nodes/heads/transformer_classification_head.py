from torch import Tensor, nn

from luxonis_train.nodes.heads import BaseHead
from luxonis_train.tasks import Tasks


class TransformerClassificationHead(BaseHead):
    """Classification decoder head for patch sequence from DINOv3.

    Converts [B, N, C] to segmentation map [B, n_classes, H, W]
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

    def forward(self, inputs: Tensor) -> Tensor:
        """
        @param inputs: CLS tokens of shape [B, C]
        @return: Class logits [B, n_classes]

        @note: Steps performed:
            1) Apply dropout to the CLS token
            2) Apply a linear layer to produce class logits.
        """
        x = self.dropout(inputs)
        return self.fc(x)
