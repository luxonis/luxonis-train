from luxonis_ml.typing import Params
from typing_extensions import override

from .base_predefined_model import SimplePredefinedModel


class DetectionModel(SimplePredefinedModel):
    def __init__(self, **kwargs):
        super().__init__(
            **{
                "backbone": "EfficientRep",
                "neck": "RepPANNeck",
                "head": "EfficientBBoxHead",
                "loss": "AdaptiveDetectionLoss",
                "metrics": "MeanAveragePrecision",
                "confusion_matrix_available": True,
                "visualizer": "BBoxVisualizer",
            }
            | kwargs
        )

    @staticmethod
    @override
    def get_variants() -> tuple[str, dict[str, Params]]:
        return "light", {
            "light": {
                "backbone_params": {
                    "variant": "n",
                    "weights": "download",
                },
                "neck_params": {
                    "variant": "n",
                    "weights": "download",
                },
            },
            "medium": {
                "backbone_params": {
                    "variant": "s",
                    "weights": "download",
                },
                "neck_params": {
                    "variant": "s",
                    "weights": "download",
                },
            },
            "heavy": {
                "backbone_params": {
                    "variant": "l",
                    "weights": "download",
                },
                "neck_params": {
                    "variant": "l",
                    "weights": "download",
                },
            },
        }
