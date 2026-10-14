"""The classification model."""

from luxonis_ml.typing import Params
from typing_extensions import override

from luxonis_train.config.predefined_models.base_predefined_model import (
    SimplePredefinedModel,
)


class ClassificationModel(SimplePredefinedModel):
    """Single-label or multi-label classification.

    Throughput:
        Frames per second at 384x512.

        - ``light``: 46 on RVC2, 176 on RVC4
        - ``heavy``: 5 on RVC2, 134 on RVC4

    Notes:
        With ``multilabel: true``, an image can have more than one
        class. The head gets the ``multilabel`` flag, and the attached
        modules follow it: `CrossEntropyLoss` computes the binary cross
        entropy of each class, `F1Score`, `Accuracy`, and `Recall` use
        the ``"multilabel"`` task of ``torchmetrics``, `ConfusionMatrix`
        gives one ``[2, 2]`` matrix for each class, and
        `ClassificationVisualizer` lists every class with a positive
        logit.

        `F1Score`, `Accuracy`, and `Recall` wrap ``torchmetrics`` and
        read a ``task`` key from ``metrics_params``. Without ``task``
        and ``num_labels``, a metric takes ``"binary"`` for one class
        and ``"multiclass"`` otherwise, and logs a warning.

    Example:
        The ``model`` section of a config:

        .. code-block:: yaml

            model:
              predefined_model:
                name: ClassificationModel
                params:
                  variant: light
                  multilabel: true

    Components:
        - Nodes: `ResNet` -> `ClassificationHead`
        - Losses: `CrossEntropyLoss`
        - Metrics:

          - `Accuracy`
          - `ConfusionMatrix`
          - `F1Score`
          - `Recall`

        - Visualizers: `ClassificationVisualizer`
        - Main metric: `F1Score`
        - Variants:

          - ``light``
          - ``heavy``

    """

    def __init__(self, multilabel: bool = False, **kwargs):
        """Initialize the model with its default components.

        Args:
            multilabel: Whether an image can have more than one class.
                ``True`` sets ``multilabel`` in ``head_params``.
            **kwargs: Keyword arguments for
                `SimplePredefinedModel.__init__`.

        """
        super().__init__(
            **{
                "backbone": "ResNet",
                "head": "ClassificationHead",
                "loss": "CrossEntropyLoss",
                "metrics": ["F1Score", "Accuracy", "Recall"],
                "confusion_matrix_available": True,
                "main_metric": "F1Score",
                "visualizer": "ClassificationVisualizer",
            }
            | kwargs
        )
        if multilabel:
            self._head_params["multilabel"] = True

    @staticmethod
    @override
    def get_variants() -> tuple[str, dict[str, Params]]:
        """Get the default variant name and the available variants.

        The default is ``light``. Both variants set ``backbone`` to
        ``"ResNet"``. They differ in ``backbone_variant``: ``"18"`` for
        ``light`` and ``"50"`` for ``heavy``.

        Returns:
            ``"light"`` and the two variants with their constructor
            arguments.

        Example:
            >>> default, variants = ClassificationModel.get_variants()
            >>> default
            'light'
            >>> variants["heavy"]
            {'backbone': 'ResNet', 'backbone_variant': '50'}

        """
        return "light", {
            "light": {
                "backbone": "ResNet",
                "backbone_variant": "18",
            },
            "heavy": {
                "backbone": "ResNet",
                "backbone_variant": "50",
            },
        }
