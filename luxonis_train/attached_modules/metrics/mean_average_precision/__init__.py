"""Mean average precision for boxes, masks, and keypoints.

`MeanAveragePrecision` is a factory. It reads the task of a node and
returns a `MeanAveragePrecisionBBox`, a
`MeanAveragePrecisionSegmentation`, or a `MeanAveragePrecisionKeypoints`
for that node. A config names the factory, and the factory selects the
metric.

"""

from .mean_average_precision import MeanAveragePrecision
from .mean_average_precision_bbox import MeanAveragePrecisionBBox
from .mean_average_precision_keypoints import MeanAveragePrecisionKeypoints
from .mean_average_precision_segmentation import (
    MeanAveragePrecisionSegmentation,
)

__all__ = [
    "MeanAveragePrecision",
    "MeanAveragePrecisionBBox",
    "MeanAveragePrecisionKeypoints",
    "MeanAveragePrecisionSegmentation",
]
