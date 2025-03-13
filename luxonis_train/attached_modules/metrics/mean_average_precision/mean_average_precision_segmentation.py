from torch import Tensor
from torchmetrics.detection import MeanAveragePrecision
from typing_extensions import override

from luxonis_train.attached_modules.metrics import BaseMetric
from luxonis_train.tasks import Tasks

from .utils import compute_update_lists, postprocess_metrics


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
        output_list, label_list = compute_update_lists(
            boundingbox, target_boundingbox, *self.original_in_shape[1:]
        )

        for i in range(len(instance_segmentation)):
            output_list[i]["masks"] = instance_segmentation[i].bool()
            label_list[i]["masks"] = target_instance_segmentation[
                target_boundingbox[:, 0] == i
            ].bool()

        super().update(output_list, label_list)

    @override
    def compute(self) -> tuple[Tensor, dict[str, Tensor]]:
        return postprocess_metrics(
            super().compute(), self.classes.inverse, "segm_map", self.device
        )
