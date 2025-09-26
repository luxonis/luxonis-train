from luxonis_ml.typing import Params
from typing_extensions import override

from .base_predefined_model import SimplePredefinedModel


class ClassificationModel(SimplePredefinedModel):
    def __init__(self, **kwargs):
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

    @staticmethod
    @override
    def get_variants() -> tuple[str, dict[str, Params]]:
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
