from torch import Tensor
from torchmetrics.classification import (
    BinaryConfusionMatrix,
    MulticlassConfusionMatrix,
)
from typing_extensions import override

from luxonis_train.attached_modules.metrics import BaseMetric
from luxonis_train.nodes.base_node import BaseNode
from luxonis_train.tasks import Tasks


class RecognitionConfusionMatrix:
    """Factory class for Recognition Confusion Matrix metrics.

    Creates the appropriate confusion matrix metric based on the number
    of classes of the node.
    """

    def __new__(cls, node: BaseNode, **kwargs) -> BaseMetric:
        n_classes = node.n_classes
        if n_classes == 1:
            return BinaryRecognitionConfusionMatrix(node=node, **kwargs)
        return MulticlassRecognitionConfusionMatrix(
            node=node, num_classes=n_classes, **kwargs
        )


class _BaseRecognitionConfusionMatrix(BaseMetric):
    """Base class for shared Recognition Confusion Matrix behavior."""

    supported_tasks = [Tasks.CLASSIFICATION, Tasks.SEGMENTATION]

    @override
    def update(self, predictions: Tensor, targets: Tensor) -> None:
        super().update(*self.preprocess(predictions, targets))

    @override
    def compute(self) -> Tensor:
        return super().compute()  # type: ignore

    def preprocess(
        self, predictions: Tensor, targets: Tensor
    ) -> tuple[Tensor, Tensor]:
        return predictions, targets


class MulticlassRecognitionConfusionMatrix(
    _BaseRecognitionConfusionMatrix, MulticlassConfusionMatrix
):
    """Multiclass specialization of RecognitionConfusionMatrix."""

    @override
    def preprocess(
        self, predictions: Tensor, targets: Tensor
    ) -> tuple[Tensor, Tensor]:
        return predictions.argmax(dim=1), targets.argmax(dim=1)


class BinaryRecognitionConfusionMatrix(
    _BaseRecognitionConfusionMatrix, BinaryConfusionMatrix
):
    """Binary specialization of RecognitionConfusionMatrix."""
