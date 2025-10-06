from loguru import logger
from luxonis_ml.typing import Params
from torch import Size, Tensor, nn
from typing_extensions import override

from luxonis_train.nodes.heads import BaseHead
from luxonis_train.tasks import Tasks


class TransformerClassificationHead(BaseHead):
    in_sizes: Size
    task = Tasks.CLASSIFICATION
    parser: str = "ClassificationParser"

    def __init__(
        self, dropout_rate: float = 0.2, use_cls_token: bool = False, **kwargs
    ):
        """Classification head for transformer patch embeddings.

        @param dropout_rate: Dropout rate before last layer.
        @param use_cls_token: If True, use the first token (CLS token)
            instead of pooling across patches.
        """
        super().__init__(**kwargs)
        self.use_cls_token = use_cls_token

        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(self.in_channels, self.n_classes)

        if len(self.input_shapes[0]["features"]) == 4:
            logger.warning(
                "The transformer segmentation head will not work with feature maps of dimension [B, C, H, W] as input. Please provide patch-level embeddings from transformer backbones in the format [B, C, N]"
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
        @param inputs: Patch embeddings of shape [B, N_patches, C]
        @return: Class logits of shape [B, n_classes]
        """
        x = inputs[:, 0] if self.use_cls_token else inputs.mean(dim=1)

        x = self.dropout(x)
        return self.fc(x)

    @override
    def get_custom_head_config(self) -> Params:
        return {"is_softmax": False}
