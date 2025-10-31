from torch import Tensor
from torchmetrics.detection import MeanAveragePrecision
from typing_extensions import override

from luxonis_train.attached_modules.metrics import BaseMetric
from luxonis_train.tasks import Tasks

from .utils import compute_metric_lists, postprocess_metrics


class MeanAveragePrecisionSegmentation(MeanAveragePrecision, BaseMetric):
    supported_tasks = [Tasks.INSTANCE_SEGMENTATION]

    def __init__(self, **kwargs):
        super().__init__(iou_type=("bbox", "segm"), **kwargs)

    @override
    def update(
        self,
        boundingbox: list[Tensor],
        instance_segmentation: list[Tensor],
        target_boundingbox: Tensor,
        target_instance_segmentation: Tensor,
    ) -> None:
        super().update(
            *compute_metric_lists(
                boundingbox,
                target_boundingbox,
                *self.original_in_shape[1:],
                masks=instance_segmentation,
                target_masks=target_instance_segmentation,
            )
        )

    @override
    def compute(self) -> tuple[Tensor, dict[str, Tensor]]:
        metrics = {k: v.to(self.device) for k, v in super().compute().items()}
        return postprocess_metrics(
            metrics, self.classes.inverse, "segm_map", self.device
        )
