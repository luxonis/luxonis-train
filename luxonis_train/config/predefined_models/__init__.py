"""Model graphs built from a name and a handful of parameters.

Name one in the ``model.predefined_model`` section instead of listing
nodes. The model then contributes the backbone, the neck, the head, and
the attached modules, and ``include_losses``, ``include_metrics``, and
``include_visualizers`` control which of those it adds.

Most models offer a ``light``, a ``medium``, and a ``heavy`` variant.
``light`` favours speed and ``heavy`` favours accuracy. Each model
docstring lists its components, its variants, and its measured
throughput.

A predefined model is versioned. ``version`` selects it, and ``latest``
is the default. A change that would alter the graph of an existing model
goes into a new version directory, so an old config keeps producing the
graph it used to.

"""

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
