"""Confusion matrices for each supported task.

A config can name the factory `ConfusionMatrix`. It builds a metric of
the matrix class that the task of the node selects. A config can also
name one of the four matrix classes directly. Each matrix also reports
the Matthews correlation coefficient (MCC), see `compute_mcc`.

Rows hold the target classes and columns hold the predicted classes, as
in ``torchmetrics``. The box matrices add a background row and column at
index ``n_classes``. The background row counts the predictions that
match no target box. The background column counts the target boxes that
match no prediction. The cell ``[n_classes, n_classes]`` counts the
images without target boxes and without predictions.

Each result is a dictionary. The trainer logs each 2D value of it as a
matrix and each scalar value as a metric. When
``trainer.log_sub_metrics`` is ``False``, the trainer logs no value of a
confusion matrix. When a confusion matrix is the main metric, the
trainer monitors the value with the name ``mcc``. The result of
`InstanceSegmentationConfusionMatrix` has no such value.

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
