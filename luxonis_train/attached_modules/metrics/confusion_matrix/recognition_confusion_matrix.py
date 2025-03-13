from torch import Tensor
from torchmetrics.classification import ConfusionMatrix
from typing_extensions import override

from luxonis_train.attached_modules.metrics import BaseMetric
from luxonis_train.tasks import Tasks


class RecognitionConfusionMatrix(BaseMetric):
    supported_tasks = [Tasks.CLASSIFICATION, Tasks.SEGMENTATION]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.confusion_matrix = ConfusionMatrix(
            num_classes=self.n_classes,
            task="binary" if self.n_classes == 1 else "multiclass",
        )

    @override
    def update(self, predictions: Tensor, targets: Tensor) -> None:
        predictions = (
            predictions.argmax(dim=1)
            if predictions.shape[1] > 1
            else predictions.squeeze(1).sigmoid().round().int()
        ).view(-1)

        targets = (
            targets.argmax(dim=1)
            if targets.shape[1] > 1
            else targets.squeeze(1).round().int()
        ).view(-1)

        self.confusion_matrix.update(predictions, targets)

    @override
    def compute(self) -> dict[str, Tensor]:
        return {
            f"{self.task.name}_confusion_matrix": self.confusion_matrix.compute()
        }

    @override
    def reset(self) -> None:
        super().reset()
        self.confusion_matrix.reset()
