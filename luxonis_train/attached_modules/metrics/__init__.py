"""Metrics that compare the predictions of a node with the labels.

A metric attaches to a node through the ``metrics`` list of the node in
the config. The trainer updates each metric on every validation and
test batch. At the end of the epoch, it computes and resets each metric
and logs the results.

Mark one metric with ``is_main_metric`` in the config. When no metric
sets it, the config marks the first metric. The trainer keeps the
checkpoints with the highest values of the main metric in the
``best_val_metric`` directory.

`MeanAveragePrecision` and `DetectionConfusionMatrix` consume boxes
after the head's non-maximum suppression, so its ``conf_thres`` and
``iou_thres`` affect their results. `PrecisionRecallCurve` uses the raw
boxes and performs its own suppression.

To write a new metric, subclass `BaseMetric`. The example of
`MetricState` shows a complete subclass.

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
