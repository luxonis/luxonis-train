from loguru import logger
from torch import Tensor
from typing_extensions import override

from luxonis_train.tasks import Tasks
from luxonis_train.utils import keypoints_to_bboxes

from .detection_confusion_matrix import DetectionConfusionMatrix


class FomoConfusionMatrix(DetectionConfusionMatrix):
    supported_tasks = [Tasks.FOMO]

    def __init__(self, iou_threshold: float | None = None, **kwargs):
        if iou_threshold is not None and iou_threshold != 0.0:
            logger.warning(
                "The `iou_threshold` parameter is ignored for FomoConfusionMatrix and is hardcoded to 0. "
                "This is by design to align with FOMO's use case.",
            )
        super().__init__(iou_threshold=0.0, **kwargs)

    @override
    def update(
        self,
        keypoints: list[Tensor],
        target_boundingbox: Tensor,
    ) -> None:
        """Override update to convert FOMO keypoints into bounding boxes
        before calling the parent update method."""
        pred_bboxes = keypoints_to_bboxes(
            keypoints,
            self.original_in_shape[1],
            self.original_in_shape[2],
        )

        super().update(pred_bboxes, target_boundingbox)
