"""The instance segmentation model."""

from luxonis_ml.typing import Params
from typing_extensions import override

from luxonis_train.config.predefined_models.base_predefined_model import (
    SimplePredefinedModel,
)


class InstanceSegmentationModel(SimplePredefinedModel):
    """Instance segmentation, which predicts a mask for each detected
    object.

    Throughput:
        Frames per second at 384x512.

        - ``light``: 39 on RVC2, 223 on RVC4
        - ``medium``: 19 on RVC2, 184 on RVC4
        - ``heavy``: 7 on RVC2, 107 on RVC4

    Example:
        The ``model`` section of a config:

        .. code-block:: yaml

            model:
              predefined_model:
                name: InstanceSegmentationModel
                params:
                  variant: light

    Components:
        - Nodes: `EfficientRep` -> `RepPANNeck` -> `PrecisionSegmentBBoxHead`
        - Losses: `PrecisionDFLSegmentationLoss`
        - Metrics:

          - `ConfusionMatrix`
          - `MeanAveragePrecision`

        - Visualizers: `InstanceSegmentationVisualizer`
        - Main metric: `MeanAveragePrecision`
        - Variants:

          - ``light``
          - ``medium``
          - ``heavy``

    """

    def __init__(self, **kwargs):
        super().__init__(
            **{
                "backbone": "EfficientRep",
                "neck": "RepPANNeck",
                "head": "PrecisionSegmentBBoxHead",
                "loss": "PrecisionDFLSegmentationLoss",
                "metrics": "MeanAveragePrecision",
                "confusion_matrix_available": True,
                "visualizer": "InstanceSegmentationVisualizer",
            }
            | kwargs
        )

    @staticmethod
    @override
    def get_variants() -> tuple[str, dict[str, Params]]:
        """Get the default variant name and the available variants.

        The default is ``light``. Each variant sets ``backbone_variant``
        and ``neck_variant`` to one size:

        - ``light``: ``"n"``
        - ``medium``: ``"s"``
        - ``heavy``: ``"l"``

        Each variant also sets ``weights`` to ``"download"`` in
        ``backbone_params`` and ``neck_params``. Both nodes then
        download and load the COCO checkpoint of their variant. The
        variants do not set ``head_params``, so the head starts without a
        checkpoint. A ``backbone_params`` or ``neck_params`` given in the
        config replaces the whole dictionary of the variant. Set
        ``weights`` in it again to keep the COCO checkpoint.

        Returns:
            ``tuple[str, dict[str, Params]]``: ``"light"`` and the three
            variants with their constructor arguments.

        Example:
            >>> default, variants = InstanceSegmentationModel.get_variants()
            >>> default
            'light'
            >>> variants["heavy"]["neck_variant"]
            'l'
            >>> variants["heavy"]["neck_params"]
            {'weights': 'download'}

        """
        return "light", {
            "light": {
                "backbone_params": {"weights": "download"},
                "backbone_variant": "n",
                "neck_params": {"weights": "download"},
                "neck_variant": "n",
            },
            "medium": {
                "backbone_params": {"weights": "download"},
                "backbone_variant": "s",
                "neck_params": {"weights": "download"},
                "neck_variant": "s",
            },
            "heavy": {
                "backbone_variant": "l",
                "backbone_params": {"weights": "download"},
                "neck_variant": "l",
                "neck_params": {"weights": "download"},
            },
        }
