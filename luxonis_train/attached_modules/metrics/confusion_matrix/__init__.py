"""Confusion matrices for each supported task.

Rows hold the ground truth and columns hold the prediction, as in
``torchmetrics``. The detection matrices add a background row and column
at index ``n_classes``, because a prediction can match no label and a
label can match no prediction.

Use a confusion matrix to find the classes a model confuses. Do not use
it as the main metric.

"""

from .confusion_matrix import ConfusionMatrix
from .detection_confusion_matrix import DetectionConfusionMatrix
from .fomo_confusion_matrix import FomoConfusionMatrix
from .instance_segmentation_confusion_matrix import (
    InstanceSegmentationConfusionMatrix,
)
from .recognition_confusion_matrix import RecognitionConfusionMatrix

__all__ = [
    "ConfusionMatrix",
    "DetectionConfusionMatrix",
    "FomoConfusionMatrix",
    "InstanceSegmentationConfusionMatrix",
    "RecognitionConfusionMatrix",
]
