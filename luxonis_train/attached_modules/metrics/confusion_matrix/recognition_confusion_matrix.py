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


class BaseRecognitionConfusionMatrix(BaseMetric):
    """Base class for shared Recognition Confusion Matrix behavior."""

    supported_tasks = [Tasks.CLASSIFICATION, Tasks.SEGMENTATION]

    @property
    def metric_name(self) -> str:
        return f"{self.task.name}_confusion_matrix"

    @override
    def update(self, predictions: Tensor, targets: Tensor) -> None:
        super().update(*self.preprocess(predictions, targets))

    @override
    def compute(self) -> dict[str, Tensor]:
        return {self.metric_name: super().compute()}  # type: ignore

    def preprocess(
        self, predictions: Tensor, targets: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Preprocesses the predictions and targets tensors before
        passing them to the confusion matrix computation.

        By default, this method does nothing and returns the predictions
        and targets unchanged.
        """
        return predictions, targets


class MulticlassRecognitionConfusionMatrix(
    BaseRecognitionConfusionMatrix, MulticlassConfusionMatrix
):
    """Multiclass specialization of RecognitionConfusionMatrix."""

    @override
    def preprocess(
        self, predictions: Tensor, targets: Tensor
    ) -> tuple[Tensor, Tensor]:
        return predictions.argmax(dim=1), targets.argmax(dim=1)


class BinaryRecognitionConfusionMatrix(
    BaseRecognitionConfusionMatrix, BinaryConfusionMatrix
):
    """Binary specialization of RecognitionConfusionMatrix."""
