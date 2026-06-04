from luxonis_ml.typing import Params
from torch import Tensor, nn
from typing_extensions import override

from luxonis_train.nodes.heads import BaseHead
from luxonis_train.tasks import Tasks


class TransformerClassificationHead(BaseHead):
    """Classification decoder head for CLS token output from DINOv3.

    Converts [B, C] (CLS token embedding) to [B, n_classes].

    Metadata:
        - Node type: head
        - Registry name: ``TransformerClassificationHead``
        - Task: classification
        - Attach index: ``-1``
        - Inputs: CLS token embedding tensor
        - Outputs: classification logits tensor

    Provenance:
        - Source: Internal
        - License: Project license
        - Implementation notes: Applies dropout and a linear classifier
          to transformer CLS token embeddings.

    Variants:
        - ``None``:
            - Default: yes
            - Aliases: None
            - Parameters:
                - No predefined variants.

    """

    attach_index = -1
    task = Tasks.CLASSIFICATION
    parser: str = "ClassificationParser"

    def __init__(self, dropout_rate: float = 0.2, **kwargs):
        """Classification head for transformer CLS tokens.

        Args:
            dropout_rate (float): Dropout rate before last layer. Defaults to ``0.2``.
            **kwargs (Any): Keyword arguments forwarded to the parent class.

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
        """Classify transformer CLS token embeddings.

        Args:
            x (Tensor): CLS tensor in the form [B, C], where C is the embedding dim.

        Returns:
            Tensor: Class logits with shape ``[B, n_classes]``.

        Notes:
            Steps performed: 1) Apply dropout to the CLS token. 2) Apply a linear layer to produce class logits.

        """
        x = self.dropout(x)
        return self.fc(x)

    @override
    def get_custom_head_config(self) -> Params:
        return {"is_softmax": False}
