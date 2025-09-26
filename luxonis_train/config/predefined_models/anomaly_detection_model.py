from luxonis_ml.typing import Params
from typing_extensions import override

from .base_predefined_model import SimplePredefinedModel


class AnomalyDetectionModel(SimplePredefinedModel):
    def __init__(self, **kwargs):
        super().__init__(
            **{
                "backbone": "RecSubNet",
                "head": "DiscSubNetHead",
                "loss": "ReconstructionSegmentationLoss",
                "metrics": "JaccardIndex",
                "confusion_matrix_available": False,
                "metrics_params": {
                    "num_classes": 2,
                    "task": "multiclass",
                },
                "visualizer": "SegmentationVisualizer",
            }
            | kwargs
        )

    @staticmethod
    @override
    def get_variants() -> tuple[str, dict[str, Params]]:
        return "light", {
            "light": {
                "backbone_variant": "n",
                "head_variant": "n",
            },
            "heavy": {
                "backbone_variant": "l",
                "head_variant": "l",
            },
        }
