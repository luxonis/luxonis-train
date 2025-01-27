import logging
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from luxonis_train.enums import Task
from luxonis_train.nodes import FOMOHead
from luxonis_train.utils.keypoints import insert_class

from .base_loss import BaseLoss

logger = logging.getLogger(__name__)


class FOMOLocalizationLoss(BaseLoss):
    node: FOMOHead
    supported_tasks = [Task.FOMO]

    def __init__(self, object_weight: float = 500, **kwargs: Any):
        """FOMO Localization Loss for object detection using heatmaps.

        @type object_weight: float
        @param object_weight: Weight for object in loss calculation.
        @type kwargs: Any
        @param kwargs: Additional arguments.
        """
        super().__init__(**kwargs)
        self.original_img_size = self.original_in_shape[1:]
        self.object_weight = object_weight

    def forward(
        self,
        heatmap: Tensor,
        target_keypoints: Tensor,
        target_boundingbox: Tensor,
    ) -> Tensor:
        target_keypoints = insert_class(target_keypoints, target_boundingbox)
        batch_size, num_classes, height, width = heatmap.shape
        target_heatmap = torch.zeros(
            (batch_size, num_classes, height, width), device=heatmap.device
        )

        for kpt in target_keypoints:
            img_idx, class_idx = int(kpt[0]), int(kpt[1])
            x_c, y_c = (
                (kpt[2] * width).round().int(),
                (kpt[3] * height).round().int(),
            )
            target_heatmap[img_idx, class_idx, y_c, x_c] = 1.0

        weight_matrix = torch.ones_like(target_heatmap)
        weight_matrix[target_heatmap == 1] = self.object_weight
        return F.binary_cross_entropy_with_logits(
            heatmap, target_heatmap, weight=weight_matrix
        )
