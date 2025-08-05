from luxonis_ml.typing import Params
from typing_extensions import override

from .base_predefined_model import SimplePredefinedModel


class KeypointDetectionModel(SimplePredefinedModel):
    def __init__(self, **kwargs):
        super().__init__(
            **{
                "backbone": "EfficientRep",
                "neck": "RepPANNeck",
                "head": "EfficientKeypointBBoxHead",
                "loss": "EfficientKeypointBBoxLoss",
                "metrics": [
                    "ObjectKeypointSimilarity",
                    "MeanAveragePrecision",
                ],
                "confusion_matrix_available": True,
                "main_metric": "MeanAveragePrecision",
                "visualizer": "KeypointVisualizer",
            }
            | kwargs
        )

    @staticmethod
    @override
    def get_variants() -> tuple[str, dict[str, Params]]:
        return "light", {
            "light": {
                "backbone_params": {"variant": "n"},
                "neck_params": {
                    "variant": "n",
                    "download_weights": True,
                },
            },
            "medium": {
                "backbone_params": {"variant": "s"},
                "neck_params": {
                    "variant": "s",
                    "download_weights": True,
                },
            },
            "heavy": {
                "backbone_params": {"variant": "l"},
                "neck_params": {
                    "variant": "l",
                    "download_weights": True,
                },
            },
        }
