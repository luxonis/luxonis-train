from luxonis_ml.typing import Params

from .base_predefined_model import SimplePredefinedModel


class InstanceSegmentationModel(SimplePredefinedModel):
    def __init__(self, **kwargs):
        kwargs = {
            "backbone": "EfficientRep",
            "neck": "RepPANNeck",
            "head": "PrecisionSegmentBBoxHead",
            "loss": "PrecisionDFLSegmentationLoss",
            "metrics": "MeanAveragePrecision",
            "confusion_matrix_available": True,
            "visualizer": "InstanceSegmentationVisualizer",
        } | kwargs
        super().__init__(**kwargs)

    @staticmethod
    def get_variants() -> tuple[str, dict[str, Params]]:
        return "light", {
            "light": {
                "backbone_params": {"variant": "n"},
                "neck_params": {"variant": "n"},
            },
            "medium": {
                "backbone_params": {"variant": "s"},
                "neck_params": {"variant": "s"},
            },
            "heavy": {
                "backbone_params": {"variant": "l"},
                "neck_params": {"variant": "l"},
            },
        }
