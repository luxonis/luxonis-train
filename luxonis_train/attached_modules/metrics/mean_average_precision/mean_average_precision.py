from luxonis_train.attached_modules.metrics import BaseMetric
from luxonis_train.nodes import BaseNode
from luxonis_train.tasks import Tasks

from .mean_average_precision_bbox import MeanAveragePrecisionBBox
from .mean_average_precision_keypoints import MeanAveragePrecisionKeypoints
from .mean_average_precision_segmentation import (
    MeanAveragePrecisionSegmentation,
)


class MeanAveragePrecision(BaseMetric):
    """Factory class for Mean Average Precision (mAP) metrics.

    Creates the appropriate mAP metric based on the task of the node.
    """

    def __new__(cls, node: BaseNode, **kwargs) -> BaseMetric:
        match node.task:
            case None:
                raise ValueError(
                    f"Node {node.name} does not have the 'task' parameter set"
                )
            case Tasks.BOUNDINGBOX:
                return MeanAveragePrecisionBBox(
                    node=node, backend="faster_coco_eval", **kwargs
                )
            case Tasks.INSTANCE_SEGMENTATION:
                return MeanAveragePrecisionSegmentation(
                    node=node, backend="faster_coco_eval", **kwargs
                )
            case Tasks.INSTANCE_KEYPOINTS:
                return MeanAveragePrecisionKeypoints(node=node, **kwargs)
            case _:
                raise ValueError(
                    f"'MeanAveragePrecision' does not support task '{node.task.name}'"
                )
