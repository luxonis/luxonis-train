from torch import Tensor

from luxonis_train.attached_modules.metrics.base_metric import BaseMetric
from luxonis_train.nodes.base_node import BaseNode
from luxonis_train.tasks import Tasks

from .detection_confusion_matrix import DetectionConfusionMatrix
from .recognition_confusion_matrix import (
    BinaryRecognitionConfusionMatrix,
    MulticlassRecognitionConfusionMatrix,
    _BaseRecognitionConfusionMatrix,
)
from .utils import preprocess_instance_masks


class InstanceSegmentationConfusionMatrix(BaseMetric):
    def __new__(cls, node: BaseNode, **kwargs) -> BaseMetric:
        n_classes = node.n_classes
        if n_classes == 1:
            return BinaryInstanceSegmentationConfusionMatrix(
                node=node, **kwargs
            )
        return MulticlassInstanceSegmentationConfusionMatrix(
            node=node, num_classes=n_classes, **kwargs
        )


class _BaseInstanceSegmentationConfusionMatrix(DetectionConfusionMatrix):
    supported_tasks = [Tasks.INSTANCE_SEGMENTATION]

    def update(
        self,
        boundingbox: list[Tensor],
        instance_segmentation: list[Tensor],
        target_boundingbox: Tensor,
        target_instance_segmentation: Tensor,
    ) -> None:
        DetectionConfusionMatrix.update(self, boundingbox, target_boundingbox)
        self.RecognitionMatrix.update(
            self,  # type: ignore
            *preprocess_instance_masks(
                boundingbox,
                instance_segmentation,
                target_boundingbox,
                target_instance_segmentation,
                self.n_classes,
                *self.original_in_shape[1:],
                device=self.device,
            ),
        )

    def compute(
        self,
    ) -> dict[str, Tensor]:
        return {
            "detection": DetectionConfusionMatrix.compute(self),
            "segmentation": self.RecognitionMatrix.compute(self),
        }

    @property
    def RecognitionMatrix(
        self,
    ) -> type[_BaseRecognitionConfusionMatrix]:
        for base in self.__class__.__bases__:
            if issubclass(base, _BaseRecognitionConfusionMatrix):
                return base
        raise RuntimeError("Internal error: no base recognition matrix found.")


class MulticlassInstanceSegmentationConfusionMatrix(
    _BaseInstanceSegmentationConfusionMatrix,
    MulticlassRecognitionConfusionMatrix,
):
    """Multiclass specialization of
    InstanceSegmentationConfusionMatrix."""


class BinaryInstanceSegmentationConfusionMatrix(
    _BaseInstanceSegmentationConfusionMatrix,
    BinaryRecognitionConfusionMatrix,
):
    """Binary specialization of InstanceSegmentationConfusionMatrix."""
