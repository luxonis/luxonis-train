from luxonis_train.nodes.base_node import BaseNode
from luxonis_train.registry import METRICS
from luxonis_train.tasks import Tasks

from .detection_confusion_matrix import DetectionConfusionMatrix
from .fomo_confusion_matrix import FomoConfusionMatrix
from .instance_segmentation_confusion_matrix import (
    InstanceSegmentationConfusionMatrix,
)
from .recognition_confusion_matrix import RecognitionConfusionMatrix


@METRICS.register()  # type: ignore
class ConfusionMatrix:
    """Factory class for Confusion Matrix metrics.

    Creates the appropriate Confusion Matrix based on the task of the
    node.
    """

    def __new__(
        cls, node: BaseNode, **kwargs
    ) -> (
        RecognitionConfusionMatrix
        | DetectionConfusionMatrix
        | InstanceSegmentationConfusionMatrix
    ):
        match node.task:
            case None:  # pragma: no cover
                raise ValueError(
                    f"Node {node.name} does not have the 'task' parameter set"
                )
            case Tasks.CLASSIFICATION | Tasks.SEGMENTATION:
                return RecognitionConfusionMatrix(node=node, **kwargs)
            case Tasks.BOUNDINGBOX | Tasks.INSTANCE_KEYPOINTS:
                return DetectionConfusionMatrix(node=node, **kwargs)
            case Tasks.INSTANCE_SEGMENTATION:
                return InstanceSegmentationConfusionMatrix(node=node, **kwargs)
            case Tasks.FOMO:
                return FomoConfusionMatrix(node=node, **kwargs)
            case _:  # pragma: no cover
                raise ValueError(
                    f"'ConfusionMatrix' does not support task '{node.task.name}'"
                )
