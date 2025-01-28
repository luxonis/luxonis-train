from luxonis_train.nodes import BaseNode
from luxonis_train.tasks import Tasks

from ..base_metric import BaseMetric
from .mean_average_precision_bbox import MeanAveragePrecisionBBox
from .mean_average_precision_keypoints import MeanAveragePrecisionKeypoints
from .mean_average_precision_segmentation import (
    MeanAveragePrecisionSegmentation,
)


class MeanAveragePrecision(BaseMetric):
    def __new__(cls, node: BaseNode, **kwargs):
        if node.task is None:
            raise ValueError(
                f"Node {node.name} does not have the 'task' parameter set."
            )
        if node.task == Tasks.BOUNDINGBOX:
            return MeanAveragePrecisionBBox(node=node, **kwargs)
        if node.task == Tasks.INSTANCE_SEGMENTATION:
            return MeanAveragePrecisionSegmentation(node=node, **kwargs)
        if node.task == Tasks.INSTANCE_KEYPOINTS:
            return MeanAveragePrecisionKeypoints(node=node, **kwargs)
        raise ValueError(
            f"MeanAveragePrecision does not support task {node.task}."
        )
