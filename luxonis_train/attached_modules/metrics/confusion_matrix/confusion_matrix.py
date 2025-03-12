from luxonis_train.attached_modules.metrics import BaseMetric
from luxonis_train.nodes.base_node import BaseNode
from luxonis_train.tasks import Tasks

from .detection_confusion_matrix import DetectionConfusionMatrix
from .recognition_confusion_matrix import RecognitionConfusionMatrix


class ConfusionMatrix(BaseMetric):
    def __new__(cls, node: BaseNode, **kwargs) -> BaseMetric:
        match node.task:
            case None:
                raise ValueError(
                    f"Node {node.name} does not have the 'task' parameter set."
                )
            case (
                Tasks.CLASSIFICATION
                | Tasks.SEGMENTATION
                | Tasks.INSTANCE_SEGMENTATION
            ):
                return RecognitionConfusionMatrix(node=node, **kwargs)
            case Tasks.BOUNDINGBOX | Tasks.INSTANCE_KEYPOINTS:
                return DetectionConfusionMatrix(node=node, **kwargs)
            case _:
                raise ValueError(
                    f"ConfusionMatrix does not support task {node.task}."
                )
