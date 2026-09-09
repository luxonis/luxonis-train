"""Mean average precision for boxes, masks, and keypoints.

`MeanAveragePrecision` is a factory. It reads the task of the node it
attaches to and returns `MeanAveragePrecisionBBox`,
`MeanAveragePrecisionSegmentation`, or `MeanAveragePrecisionKeypoints`.
Name the factory in a config and let it choose.

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
