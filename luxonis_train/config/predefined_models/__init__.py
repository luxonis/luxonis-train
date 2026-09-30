"""Model graphs built from a name and a few parameters.

Name a model in the ``model.predefined_model`` section of a config. The
config then appends the nodes of the model to ``model.nodes``, after the
nodes that the config lists. These nodes carry the losses, the metrics,
and the visualizers of the model. ``include_losses``,
``include_metrics``, and ``include_visualizers`` control which of them
the nodes keep.

Most models offer a ``light`` and a ``heavy`` variant, and some offer a
``medium`` one between them. ``light`` is the fastest and ``heavy`` is
the most accurate. Each model docstring lists its components and its
variants, and most also list the measured throughput.

A predefined model is versioned. ``version`` selects the version, and
``"latest"`` is the default. A change that alters the graph of an
existing model goes into a new version directory. A config that pins
``version`` therefore still builds the same graph.

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
