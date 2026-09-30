"""A classification head over the CLS token of a transformer
backbone.
"""

from luxonis_ml.typing import Params
from torch import Tensor, nn
from typing_extensions import override

from luxonis_train.nodes.heads import BaseHead
from luxonis_train.tasks import Tasks


class TransformerClassificationHead(BaseHead):
    r"""Classification head for the CLS token of a transformer backbone.

    `DinoV3` gives the CLS token when its ``return_sequence`` param is
    ``True``.

    Inputs:
        - ``inputs`` (``Tensor``): :math:`\left[B, C\right]` CLS token

    Outputs:
        - ``classification`` (``Tensor``): :math:`\left[B,
          n_{classes}\right]` logits

    References:
        - Source: This project.
        - License: Apache-2.0 (this project)

    Notes:
        The head applies dropout and one linear layer to the CLS token.
        It has no pooling, because the CLS token has no spatial
        dimensions. The dropout acts only in training mode. The mode
        does not change the output key or shape. This includes export
        mode.

    Variants:
        None. Configure the node through ``params``.

    Example:
        A node entry in the ``model.nodes`` section of a config:

        .. code-block:: yaml

            - name: TransformerClassificationHead
              inputs: [DinoV3]

    Compatible with:
        - Attach index: ``-1``, the last output of the input node
        - Required labels: ``classification``
        - Losses:

          - `BCEWithLogitsLoss`
          - `CrossEntropyLoss`
          - `OHEMLoss`
          - `SigmoidFocalLoss`
          - `SmoothBCEWithLogitsLoss`
          - `SoftmaxFocalLoss`

        - Metrics:

          - `Accuracy`
          - `ConfusionMatrix`
          - `F1Score`
          - `JaccardIndex`
          - `Precision`
          - `Recall`

        - Visualizers: `ClassificationVisualizer`
        - Export parser: ``ClassificationParser``

    """

    attach_index = -1
    task = Tasks.CLASSIFICATION
    parser: str = "ClassificationParser"

    def __init__(self, dropout_rate: float = 0.2, **kwargs):
        """Build the dropout and the linear layer.

        The linear layer maps `in_channels` values to ``n_classes``
        logits.

        Args:
            dropout_rate (float): The probability that the dropout layer
                sets a value of the CLS token to zero in training mode,
                in ``[0, 1]``. The layer scales the other values by
                :math:`1 / (1 - p)`, where :math:`p` is
                ``dropout_rate``. Defaults to ``0.2``.
            **kwargs (``Any``): Keyword arguments for `BaseNode`. They
                must hold the input sizes through ``input_shapes`` or
                ``in_sizes``, and the class count through ``n_classes``
                or ``dataset_metadata``.

        """
        super().__init__(**kwargs)

        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(self.in_channels, self.n_classes)

    @property
    def in_channels(self) -> int:
        """The embedding size of the CLS token.

        It is the last dimension of `BaseNode.in_sizes`, the ``C`` of
        the input shape ``[B, C]``. It replaces `BaseNode.in_channels`,
        which reads the third dimension from the end.

        Raises:
            TypeError: When `BaseNode.in_sizes` is a list of sizes. This
                occurs for an ``attach_index`` of ``"all"`` or a range.
                The constructor reads the property, so it raises the
                error too.

        """
        result = self._get_nth_size(-1)
        if isinstance(result, list):
            raise TypeError("Expected a single [B, C], got multiple.")
        return result

    def forward(self, x: Tensor) -> Tensor:
        """Compute the class logits from the CLS token.

        The method applies the dropout and then the linear layer. The
        dropout acts only in training mode.

        Args:
            x (``Tensor``): The CLS token of shape ``[B, C]``, where ``C``
                is the embedding size.

        Returns:
            ``Tensor``: The logits of shape ``[B, n_classes]``.
            `BaseNode.run` puts them under the ``"classification"`` key.

        Example:
            >>> import torch
            >>> from torch import Size
            >>> from luxonis_train.nodes import TransformerClassificationHead
            >>> head = TransformerClassificationHead(
            ...     n_classes=5,
            ...     input_shapes=[{"features": [Size([2, 384])]}],
            ... )
            >>> head.in_channels
            384
            >>> head(torch.zeros(2, 384)).shape
            torch.Size([2, 5])

        """
        x = self.dropout(x)
        return self.fc(x)

    @override
    def get_custom_head_config(self) -> Params:
        return {"is_softmax": False}
