from luxonis_train.registry import MODELS

from .anomaly_detection.v1.model import AnomalyDetectionModel
from .base_predefined_model import BasePredefinedModel
from .classification.v1.model import ClassificationModel
from .detection.v1.model import DetectionModel
from .embeddings.v1.model import EmbeddingsModel
from .fomo.v1.model import FOMOModel
from .instance_segmentation.v1.model import InstanceSegmentationModel
from .keypoint_detection.v1.model import KeypointDetectionModel
from .ocr_recognition.v1.model import OCRRecognitionModel
from .segmentation.v1.model import SegmentationModel


def _model_family_name(cls: type[BasePredefinedModel]) -> str:
    """Return the stable registry family for a predefined model class.

    Breaking versions are named ``FamilyV2``, ``FamilyV3``, and so on,
    while users continue to address them as ``Family``.
    """
    version_suffix = f"V{cls._VERSION}"
    if cls.__name__.endswith(version_suffix):
        return cls.__name__[: -len(version_suffix)]
    return cls.__name__


def _rekey_registry_with_versions() -> None:
    """Replace plain-class-name registry keys with ``Family:vN`` keys.

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
        versioned = f"{_model_family_name(cls)}:v{cls._VERSION}"
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
    "EmbeddingsModel",
    "FOMOModel",
    "InstanceSegmentationModel",
    "KeypointDetectionModel",
    "OCRRecognitionModel",
    "SegmentationModel",
]
