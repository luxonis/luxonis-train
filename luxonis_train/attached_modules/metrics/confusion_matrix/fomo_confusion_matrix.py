from typing import Optional

import torch
from loguru import logger
from torch import Tensor
from typing_extensions import override

from luxonis_train.tasks import Tasks

from .detection_confusion_matrix import DetectionConfusionMatrix
from .utils import compute_mcc


class FomoConfusionMatrix(DetectionConfusionMatrix):
    supported_tasks = [Tasks.FOMO]

    def __init__(self, iou_threshold: Optional[float] = None, **kwargs):
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
        pred_bboxes = self._keypoints_to_bboxes(
            keypoints,
            self.original_in_shape[1],
            self.original_in_shape[2],
        )

        super().update(pred_bboxes, target_boundingbox)

    @override
    def compute(self) -> Tensor:
        return {
            "mcc": compute_mcc(self.confusion_matrix.float()),
            "confusion_matrix": self.confusion_matrix,
        }

    @staticmethod
    def _keypoints_to_bboxes(
        keypoints: list[Tensor],
        img_height: int,
        img_width: int,
        box_width: int = 5,
        visibility_threshold: float = 0.5,
    ) -> list[Tensor]:
        """Convert keypoints to bounding boxes in xyxy format with
        cls_id and score, filtering low-visibility keypoints.

        @type keypoints: list[Tensor]
        @param keypoints: List of tensors of keypoints with shape [N, 1,
            4] (x, y, v, cls_id).
        @type img_height: int
        @param img_height: Height of the image.
        @type img_width: int
        @param img_width: Width of the image.
        @type box_width: int
        @param box_width: Width of the bounding box in pixels. Defaults
            to 2.
        @type visibility_threshold: float
        @param visibility_threshold: Minimum visibility score to include
            a keypoint. Defaults to 0.5.
        @rtype: list[Tensor]
        @return: List of tensors of bounding boxes with shape [N, 6]
            (x_min, y_min, x_max, y_max, score, cls_id).
        """
        half_box = box_width / 2
        bboxes_list = []

        for keypoints_per_image in keypoints:
            if keypoints_per_image.numel() == 0:
                bboxes_list.append(
                    torch.zeros((0, 6), device=keypoints_per_image.device)
                )
                continue

            keypoints_per_image = keypoints_per_image.squeeze(1)

            visible_mask = keypoints_per_image[:, 2] >= visibility_threshold
            keypoints_per_image = keypoints_per_image[visible_mask]

            if keypoints_per_image.numel() == 0:
                bboxes_list.append(
                    torch.zeros((0, 6), device=keypoints_per_image.device)
                )
                continue

            x_coords = keypoints_per_image[:, 0]
            y_coords = keypoints_per_image[:, 1]
            scores = keypoints_per_image[:, 2]
            cls_ids = keypoints_per_image[:, 3]

            x_min = (x_coords - half_box).clamp(min=0)
            y_min = (y_coords - half_box).clamp(min=0)
            x_max = (x_coords + half_box).clamp(max=img_width)
            y_max = (y_coords + half_box).clamp(max=img_height)
            bboxes = torch.stack(
                [x_min, y_min, x_max, y_max, scores, cls_ids], dim=-1
            )
            bboxes_list.append(bboxes)

        return bboxes_list
