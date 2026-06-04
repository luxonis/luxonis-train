from loguru import logger
from torch import Tensor
from typing_extensions import override

from luxonis_train.tasks import Tasks
from luxonis_train.utils import keypoints_to_bboxes

from .detection_confusion_matrix import DetectionConfusionMatrix


class FomoConfusionMatrix(DetectionConfusionMatrix):
    """Confusion matrix for FOMO keypoint predictions.

    Metadata:
        - Module type: metric
        - Registry name: ``FomoConfusionMatrix``
        - Task: FOMO
        - Attached node types: None
        - Inputs: ``keypoints``, ``target_boundingbox``
        - Outputs: dictionary with ``mcc`` and ``confusion_matrix``
        - State: ``confusion_matrix``

    Prediction format:
        ``keypoints`` is a list of per-image FOMO keypoint detections.

    Target format:
        ``target_boundingbox`` contains batch-indexed boxes with class IDs and
        normalized ``xywh`` coordinates.

    Formula:
        Converts FOMO keypoints to zero-threshold boxes, then delegates matching
        and accumulation to ``DetectionConfusionMatrix``.

    Provenance:
        - Source: Internal
        - License: Project license
        - Implementation notes: Ignores user-provided nonzero IoU thresholds to
          preserve FOMO center-point semantics.

    """

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
        """Convert FOMO keypoints into bounding boxes before updating.

        Args:
            keypoints (list[Tensor]): Predicted FOMO keypoints.
            target_boundingbox (Tensor): Target bounding boxes.

        """
        pred_bboxes = keypoints_to_bboxes(
            keypoints,
            self.original_in_shape[1],
            self.original_in_shape[2],
        )

        super().update(pred_bboxes, target_boundingbox)
