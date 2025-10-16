from torch import Tensor
from torchmetrics.detection import MeanAveragePrecision
from typing_extensions import override

from luxonis_train.attached_modules.metrics import BaseMetric
from luxonis_train.tasks import Tasks

from .utils import compute_metric_lists, postprocess_metrics


class MeanAveragePrecisionBBox(MeanAveragePrecision, BaseMetric):
    supported_tasks = [Tasks.BOUNDINGBOX]

    def __init__(self, **kwargs):
        super().__init__(iou_type="bbox", **kwargs)

    @override
    def update(self, predictions: list[Tensor], targets: Tensor) -> None:
        super().update(
            *compute_metric_lists(
                predictions, targets, *self.original_in_shape[1:]
            )
        )

    @override
    def compute(self) -> tuple[Tensor, dict[str, Tensor]]:
        metrics = {k: v.to(self.device) for k, v in super().compute().items()}
        return postprocess_metrics(
            metrics, self.classes.inverse, "map", self.device
        )
