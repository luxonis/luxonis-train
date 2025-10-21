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
                "backbone_params": {"weights": "download"},
                "backbone_variant": "l",
                "neck_params": {"weights": "download"},
                "neck_variant": "l",
            },
        }
