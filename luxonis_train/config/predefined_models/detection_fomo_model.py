from luxonis_ml.typing import Params
from typing_extensions import override

from .base_predefined_model import SimplePredefinedModel


class FOMOModel(SimplePredefinedModel):
    def __init__(self, **kwargs):
        super().__init__(
            **{
                "backbone": "EfficientRep",
                "head": "FOMOHead",
                "loss": "FOMOLocalizationLoss",
                "metrics": "ConfusionMatrix",
                "confusion_matrix_available": True,
                "visualizer": "FOMOVisualizer",
            }
            | kwargs
        )

    @staticmethod
    @override
    def get_variants() -> tuple[str, dict[str, Params]]:
        return "light", {
            "light": {
                "backbone": "EfficientRep",
                "head_params": {
                    "n_conv_layers": 2,
                    "conv_channels": 16,
                },
                "backbone_variant": "n",
            },
            "heavy": {
                "backbone": "MobileNetV2",
                "head_params": {
                    "n_conv_layers": 2,
                    "conv_channels": 16,
                },
            },
        }
