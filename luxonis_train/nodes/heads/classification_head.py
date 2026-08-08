from luxonis_ml.typing import Params
from torch import Tensor, nn
from typing_extensions import override

from luxonis_train.nodes.heads import BaseHead
from luxonis_train.tasks import Tasks


class ClassificationHead(BaseHead):
    """Simple classification head.

    Consists of a global average pooling layer followed by a dropout
    layer and a single linear layer.
    """

    in_channels: int
    task = Tasks.CLASSIFICATION
    parser: str = "ClassificationParser"

    def __init__(self, dropout_rate: float = 0.2, **kwargs):
        """Classification head.

        Args:
            dropout_rate: Dropout rate before last layer, range
                ``[0, 1]``. Defaults to ``0.2``.
            **kwargs: Base node arguments.

        """
        super().__init__(**kwargs)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.in_channels, self.n_classes),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        r"""Pool image features and predict class logits.

        Args:
            inputs: Input feature map.

        Returns:
            Class logits for each image.

        .. shape-contract::

            Inputs
                :math:`\mathrm{inputs}`
                    :math:`(B, C_{\mathrm{in}}, H, W)`

            Outputs
                :math:`\mathrm{logits}`
                    :math:`(B, n_{\mathrm{classes}})`

            Symbols
                :math:`n_{\mathrm{classes}}`
                    Number of predicted classes.

        """
        return self.head(inputs)

    @override
    def get_custom_head_config(self) -> Params:
        return {"is_softmax": False}
