from luxonis_train.nodes import BaseNode
from luxonis_train.registry import METRICS
from luxonis_train.tasks import Tasks

from .mean_average_precision_bbox import MeanAveragePrecisionBBox
from .mean_average_precision_keypoints import MeanAveragePrecisionKeypoints
from .mean_average_precision_segmentation import (
    MeanAveragePrecisionSegmentation,
)


@METRICS.register()  # type: ignore
class MeanAveragePrecision:
    """Factory class for Mean Average Precision (mAP) metrics.

    Creates the appropriate mAP metric based on the task of the node.
    """

    def __new__(
        cls, node: BaseNode, **kwargs
    ) -> (
        MeanAveragePrecisionBBox
        | MeanAveragePrecisionSegmentation
        | MeanAveragePrecisionKeypoints
    ):
        match node.task:
            case None:  # pragma: no cover
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
            case _:  # pragma: no cover
                raise ValueError(
                    f"'MeanAveragePrecision' does not support task '{node.task.name}'"
                )
