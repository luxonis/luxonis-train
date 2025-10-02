from luxonis_ml.typing import Params
from torch import Tensor, nn
from typing_extensions import override

from luxonis_train.nodes.heads import BaseHead
from luxonis_train.tasks import Tasks


class TransformerClassificationHead(BaseHead):
    task = Tasks.CLASSIFICATION
    parser: str = "ClassificationParser"

    def __init__(self, dropout_rate: float = 0.2, use_cls_token: bool = False, **kwargs):
        """
        Classification head for transformer patch embeddings.

        @param dropout_rate: Dropout rate before last layer.
        @param use_cls_token: If True, use the first token (CLS token)
                              instead of pooling across patches.
        """
        super().__init__(**kwargs)
        self.use_cls_token = use_cls_token

        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(self.in_channels, self.n_classes)

    @property
    def in_channels(self) -> int:
        """
        Override to extract embedding dim from transformer output shape.
        Expected input_shapes: [{'features': [torch.Size([B, N, C])]}]
        """
        try:
            shape_dict = self.input_shapes[0]
            feature_shape = shape_dict["features"][0]

            return feature_shape[-1]
        except Exception as e:
            raise RuntimeError(f"Could not determine in_channels from input_shapes: {self.input_shapes} — {e}")

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Args:
            inputs: Patch embeddings of shape [B, N_patches, C]
        Returns:
            Class logits of shape [B, n_classes]
        """
        if self.use_cls_token:
            x = inputs[:, 0]  # [B, C]
        else:
            x = inputs.mean(dim=1)

        x = self.dropout(x)
        x = self.fc(x)
        return x

    @override
    def get_custom_head_config(self) -> Params:
        return {"is_softmax": False}