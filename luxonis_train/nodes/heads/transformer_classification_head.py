from loguru import logger
from torch import Size, Tensor, nn

from luxonis_train.nodes.heads import BaseHead
from luxonis_train.tasks import Tasks


class TransformerClassificationHead(BaseHead):
    """Classification decoder head for patch sequence from DINOv3.

    Converts [B, N, C] to segmentation map [B, n_classes, H, W]
    """

    in_sizes: Size
    task = Tasks.CLASSIFICATION
    parser: str = "ClassificationParser"

    def __init__(self, dropout_rate: float = 0.2, **kwargs):
        """Classification head for transformer patch embeddings.

        @param dropout_rate: Dropout rate before last layer.
        """
        super().__init__(**kwargs)

        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(self.in_channels, self.n_classes)

        if len(self.in_sizes) == 4:
            logger.warning(
                "The transformer classification head expects patch-level embeddings "
                "in the format [B, N, C], not [B, C, H, W]."
            )

    @property
    def in_channels(self) -> int:
        """Extract embedding dim from self.in_sizes instead of
        input_shapes."""
        try:
            return self.in_sizes[-1]
        except Exception as e:
            raise RuntimeError(
                f"Could not determine in_channels from in_sizes: {self.in_sizes} — {e}"
            ) from e

    def forward(self, inputs: Tensor) -> Tensor:
        """
        @param inputs: Patch embeddings of shape [B, N, C]
        @return: Class logits of shape [B, n_classes]

        @note: Steps performed:
            1) Mean pooling over patch dimensions.
            2) Apply dropout to the pooled embeddings
            3) Apply a linear layer to produce class logits.
        """
        x = inputs.mean(dim=1)
        x = self.dropout(x)
        return self.fc(x)
