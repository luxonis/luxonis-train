import logging
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from luxonis_train.enums import TaskType
from luxonis_train.nodes import FOMOHead
from luxonis_train.utils import Labels, Packet

from .base_loss import BaseLoss

logger = logging.getLogger(__name__)


class FOMOLocalizationLoss(BaseLoss[Tensor, Tensor]):
    node: FOMOHead
    supported_tasks: list[tuple[TaskType, ...]] = [
        (TaskType.BOUNDINGBOX, TaskType.KEYPOINTS)
    ]

    def __init__(self, object_weight: float = 1000, **kwargs: Any):
        """FOMO Localization Loss for object detection using heatmaps.

        @type object_weight: float
        @param object_weight: Weight for object in loss calculation.
        @type kwargs: Any
        @param kwargs: Additional arguments.
        """
        super().__init__(**kwargs)
        self.original_img_size = self.original_in_shape[1:]
        self.object_weight = object_weight

    def prepare(
        self, inputs: Packet[Tensor], labels: Labels
    ) -> tuple[Tensor, Tensor]:
        heatmap = self.get_input_tensors(inputs, "features")[0]
        target_kpts = self.get_label(labels, TaskType.KEYPOINTS)
        batch_size, num_classes, height, width = heatmap.shape
        target_heatmap = torch.zeros(
            (batch_size, num_classes, height, width), device=heatmap.device
        )

        for kpt in target_kpts:
            img_idx, class_idx = int(kpt[0]), int(kpt[1])
            x_c, y_c = (
                (kpt[2] * width).round().int(),
                (kpt[3] * height).round().int(),
            )
            target_heatmap[img_idx, class_idx, y_c, x_c] = 1.0

        return heatmap, target_heatmap

    def forward(
        self, predicted_heatmap: Tensor, target_heatmap: Tensor
    ) -> Tensor:
        """Forward pass for FOMO Localization Loss.

        @type predicted_heatmap: Tensor
        @param predicted_heatmap: Predicted heatmap.
        @type target_heatmap: Tensor
        @param target_heatmap: Target heatmap.
        @rtype: Tensor
        @return: Loss value.
        """
        weight_matrix = torch.ones_like(target_heatmap)
        weight_matrix[target_heatmap == 1] = self.object_weight
        return F.binary_cross_entropy_with_logits(
            predicted_heatmap, target_heatmap, weight=weight_matrix
        )
