from torch import Tensor
from torchmetrics.classification import ConfusionMatrix
from typing_extensions import override

from luxonis_train.attached_modules.metrics import BaseMetric
from luxonis_train.tasks import Tasks


class RecognitionConfusionMatrix(BaseMetric, register=False):
    supported_tasks = [
        Tasks.CLASSIFICATION,
        Tasks.SEGMENTATION,
        Tasks.INSTANCE_SEGMENTATION,
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.metric_cm = ConfusionMatrix(
            num_classes=self.n_classes,
            task="binary" if self.n_classes == 1 else "multiclass",
        )

    @override
    def update(self, predictions: Tensor, target: Tensor) -> None:
        predictions = (
            predictions.argmax(dim=1)
            if predictions.shape[1] > 1
            else predictions.squeeze(1).sigmoid().round().int()
        ).view(-1)

        targets = (
            target.argmax(dim=1)
            if target.shape[1] > 1
            else target.squeeze(1).round().int()
        ).view(-1)

        self.metric_cm.update(predictions, targets)

    @override
    def compute(self) -> dict[str, Tensor]:
        return {f"{self.task}_confusion_matrix": self.metric_cm.compute()}

    @override
    def reset(self) -> None:
        self.metric_cm.reset()
