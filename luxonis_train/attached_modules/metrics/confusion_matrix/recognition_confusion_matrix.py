from torch import Tensor
from torchmetrics.classification import (
    BinaryConfusionMatrix,
    MulticlassConfusionMatrix,
)
from typing_extensions import override

from luxonis_train.attached_modules.metrics import BaseMetric
from luxonis_train.tasks import Tasks

from .utils import compute_mcc


class RecognitionConfusionMatrix(BaseMetric):
    """Factory class for Recognition Confusion Matrix metrics.

    Creates the appropriate confusion matrix metric based on the number
    of classes of the node.
    """

    supported_tasks = [Tasks.CLASSIFICATION, Tasks.SEGMENTATION]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.n_classes == 1:
            self.metric = BinaryConfusionMatrix()
        else:
            self.metric = MulticlassConfusionMatrix(num_classes=self.n_classes)

    @override
    def update(self, predictions: Tensor, targets: Tensor) -> None:
        if self.n_classes > 1:
            self.metric.update(
                predictions.argmax(dim=1), targets.argmax(dim=1)
            )
        else:
            self.metric.update(predictions, targets)

    @override
    def compute(self) -> dict[str, Tensor]:
        cm = self.metric.compute()
        mcc = compute_mcc(cm.float())
        return {"mcc": mcc, "confusion_matrix": cm}

    @override
    def reset(self) -> None:
        self.metric.reset()
