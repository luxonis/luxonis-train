from torch import Tensor
from torchmetrics.detection import MeanAveragePrecision
from typing_extensions import override

from luxonis_train.attached_modules.metrics import BaseMetric
from luxonis_train.tasks import Tasks

from .utils import compute_metric_lists, postprocess_metrics


class MeanAveragePrecisionSegmentation(MeanAveragePrecision, BaseMetric):
    """Mean average precision for instance segmentation masks.

    Metadata:
        - Module type: metric
        - Registry name: ``MeanAveragePrecisionSegmentation``
        - Task: INSTANCE_SEGMENTATION, INSTANCE_SEGMENTATION_KEYPOINTS
        - Attached node types: None
        - Inputs: ``boundingbox``, ``instance_segmentation``,
          ``target_boundingbox``, ``target_instance_segmentation``
        - Outputs: main ``segm_map`` tensor and dictionary of AP/AR sub-metrics
        - State: wrapped ``torchmetrics.detection.MeanAveragePrecision`` state

    Prediction format:
        ``boundingbox`` is a list of per-image detections, and
        ``instance_segmentation`` is a list of predicted masks.

    Target format:
        ``target_boundingbox`` contains batch-indexed boxes with class IDs and
        normalized ``xywh`` coordinates. ``target_instance_segmentation``
        contains instance masks aligned to those boxes.

    Formula:
        Converts predictions and targets into torchmetrics detection lists and
        evaluates bbox and segmentation mAP/mAR.

    Provenance:
        - Source: torchmetrics
        - License: Project license
        - Implementation notes: Uses ``iou_type=("bbox", "segm")`` and
          postprocesses segmentation metrics with dataset class names.

    """

    supported_tasks = [
        Tasks.INSTANCE_SEGMENTATION,
        Tasks.INSTANCE_SEGMENTATION_KEYPOINTS,
    ]

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
