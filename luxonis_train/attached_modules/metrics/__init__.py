"""Metrics computed on the predictions of a node.

`Accuracy`, `F1Score`, `JaccardIndex`, `Precision`, and `Recall` wrap
the matching ``torchmetrics`` classes. `MIoU`, `DiceCoefficient`,
`ObjectKeypointSimilarity`, `MeanAveragePrecision`,
`PrecisionRecallCurve`, `OCRAccuracy`, `ConfusionMatrix`,
`ClosestIsPositiveAccuracy`, and `MedianDistances` cover the tasks that
``torchmetrics`` does not.

Mark one metric with ``is_main_metric`` in the config. The trainer saves
a checkpoint on that metric.

Every detection metric reads the predictions after non-maximum
suppression, so the ``conf_thres`` and ``iou_thres`` of the head change
the result. Tune both for your data.

"""

from .base_metric import BaseMetric, DistReduceFx, MetricState
from .confusion_matrix import ConfusionMatrix
from .dice_coefficient import DiceCoefficient
from .embedding_metrics import ClosestIsPositiveAccuracy, MedianDistances
from .mean_average_precision import MeanAveragePrecision
from .mean_iou import MIoU
from .object_keypoint_similarity import ObjectKeypointSimilarity
from .ocr_accuracy import OCRAccuracy
from .precision_recall_curve import PrecisionRecallCurve
from .torchmetrics import Accuracy, F1Score, JaccardIndex, Precision, Recall
from .utils import fix_empty_tensor, merge_bbox_kpt_targets

__all__ = [
    "Accuracy",
    "BaseMetric",
    "ClosestIsPositiveAccuracy",
    "ConfusionMatrix",
    "DiceCoefficient",
    "DistReduceFx",
    "F1Score",
    "JaccardIndex",
    "MIoU",
    "MeanAveragePrecision",
    "MedianDistances",
    "MetricState",
    "OCRAccuracy",
    "ObjectKeypointSimilarity",
    "Precision",
    "PrecisionRecallCurve",
    "Recall",
    "fix_empty_tensor",
    "merge_bbox_kpt_targets",
]
