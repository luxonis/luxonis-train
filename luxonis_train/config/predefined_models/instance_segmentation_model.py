from luxonis_ml.typing import Params
from typing_extensions import override

from .base_predefined_model import SimplePredefinedModel


class InstanceSegmentationModel(SimplePredefinedModel):
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
        return "light", {
            "light": {
                "backbone_variant": "n",
                "neck_variant": "n",
            },
            "medium": {
                "backbone_variant": "s",
                "neck_variant": "s",
            },
            "heavy": {
                "backbone_variant": "l",
                "neck_variant": "l",
            },
        }
