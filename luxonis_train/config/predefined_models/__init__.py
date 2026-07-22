from luxonis_train.registry import MODELS

from .anomaly_detection.v1.model import AnomalyDetectionModel
from .base_predefined_model import BasePredefinedModel
from .classification.v1.model import ClassificationModel
from .detection.v1.model import DetectionModel
from .fomo.v1.model import FOMOModel
from .instance_segmentation.v1.model import InstanceSegmentationModel
from .keypoint_detection.v1.model import KeypointDetectionModel
from .ocr_recognition.v1.model import OCRRecognitionModel
from .segmentation.v1.model import SegmentationModel


def _rekey_registry_with_versions() -> None:
    """Replace plain-class-name registry keys with ``ClassName:vN``
    keys.

    Runs after every predefined-model subclass module has been imported
    (which is what triggers ``AutoRegisterMeta`` to register the class
    under its plain ``__name__``). We swap those in-place for the
    versioned key that :mod:`luxonis_train.config.predefined_versions`
    expects.
    """
    for key, cls in list(MODELS._module_dict.items()):
        if not isinstance(cls, type) or not issubclass(
            cls, BasePredefinedModel
        ):
            continue
        # Skip abstract intermediates like ``SimplePredefinedModel``.
        if getattr(cls, "__abstractmethods__", frozenset()):
            MODELS._module_dict.pop(key, None)
            continue
        if ":" in key:
            continue  # already versioned
        versioned = f"{cls.__name__}:v{cls._VERSION}"
        if versioned in MODELS._module_dict:
            continue
        MODELS._module_dict.pop(key, None)
        MODELS._module_dict[versioned] = cls


_rekey_registry_with_versions()


__all__ = [
    "AnomalyDetectionModel",
    "BasePredefinedModel",
    "ClassificationModel",
    "DetectionModel",
    "FOMOModel",
    "InstanceSegmentationModel",
    "KeypointDetectionModel",
    "OCRRecognitionModel",
    "SegmentationModel",
]
