from torch import Tensor
from torchmetrics.detection import MeanAveragePrecision
from typing_extensions import override

from luxonis_train.attached_modules.metrics import BaseMetric
from luxonis_train.tasks import Tasks

from .utils import compute_metric_lists, postprocess_metrics


class MeanAveragePrecisionBBox(MeanAveragePrecision, BaseMetric):
    """Mean average precision for bbox detections.

    Metadata:
        - Module type: metric
        - Registry name: ``MeanAveragePrecisionBBox``
        - Task: BOUNDINGBOX, INSTANCE_KEYPOINTS, INSTANCE_SEGMENTATION,
          INSTANCE_SEGMENTATION_KEYPOINTS
        - Attached node types: None
        - Inputs: ``boundingbox``, ``target_boundingbox``
        - Outputs: main ``map`` tensor and dictionary of AP/AR sub-metrics
        - State: wrapped ``torchmetrics.detection.MeanAveragePrecision`` state

    Prediction format:
        ``boundingbox`` is a list of per-image detections with boxes, scores,
        and predicted class IDs.

    Target format:
        ``target_boundingbox`` contains batch-indexed boxes with class IDs and
        normalized ``xywh`` coordinates.

    Formula:
        Converts predictions and targets into torchmetrics detection lists and
        evaluates bbox mAP/mAR.

    Provenance:
        - Source: torchmetrics
        - License: Project license
        - Implementation notes: Uses ``iou_type="bbox"`` and postprocesses
          class metrics with dataset class names.

    """

    supported_tasks = [
        Tasks.BOUNDINGBOX,
        Tasks.INSTANCE_KEYPOINTS,
        Tasks.INSTANCE_SEGMENTATION,
        Tasks.INSTANCE_SEGMENTATION_KEYPOINTS,
    ]

    def __init__(self, **kwargs):
        super().__init__(iou_type="bbox", **kwargs)

    @override
    def update(
        self, boundingbox: list[Tensor], target_boundingbox: Tensor
    ) -> None:
        super().update(
            *compute_metric_lists(
                boundingbox,
                target_boundingbox,
                *self.original_in_shape[1:],
            )
        )

    @override
    def compute(self) -> tuple[Tensor, dict[str, Tensor]]:
        metrics = {k: v.to(self.device) for k, v in super().compute().items()}
        return postprocess_metrics(
            metrics, self.classes.inverse, "map", self.device
        )
