import logging
from typing import Any, List

import torch
from torch import Tensor

from luxonis_train.enums import TaskType
from luxonis_train.nodes.base_node import BaseNode
from luxonis_train.utils import Packet

logger = logging.getLogger(__name__)


class FOMOHead(BaseNode[list[Tensor], list[Tensor]]):
    in_channels: list[int]
    tasks: list[TaskType] = [TaskType.BOUNDINGBOX]

    def __init__(
        self,
        **kwargs: Any,
    ):
        """FOMO Head for object detection using heatmaps.

        @type kwargs: Any
        @param kwargs: Additional arguments.
        """
        super().__init__(**kwargs)
        self.original_img_size = self.original_in_shape[1:]

        self.conv1 = torch.nn.Conv2d(
            self.in_channels[2], 16, kernel_size=1, stride=1
        )  # Hardcoding in_channels[2] for now -> keeping 20x20 heatmaps
        self.conv2 = torch.nn.Conv2d(16, 16, kernel_size=1, stride=1)
        self.conv3 = torch.nn.Conv2d(
            16, self.n_classes, kernel_size=1, stride=1
        )

    def forward(self, inputs: List[Tensor]) -> Tensor:
        x = torch.relu(
            self.conv1(inputs[2])
        )  # [16,24,80,80], [16,32,40,40], [16, 96,20,20], [16, 1280,10,10] # When imgsz = 320, bs = 16
        x = torch.relu(self.conv2(x))
        heatmap = self.conv3(x)
        return heatmap

    def wrap(self, heatmap: Tensor) -> Packet[Tensor]:
        if self.training or self.export:
            return {
                "features": [heatmap],
            }
        else:
            boxes = self._heatmap_to_xyxy(heatmap)
            return {
                "boundingbox": boxes,
                "features": [heatmap],
            }

    def _heatmap_to_xyxy(self, heatmap: Tensor) -> List[Tensor]:
        """Convert heatmap to bounding boxes, ensuring all tensors are
        on the same device."""
        device = heatmap.device
        batch_size, num_classes, height, width = heatmap.shape
        bbox_size = (
            self.original_img_size[1] + self.original_img_size[0]
        ) / 40

        batch_boxes = []
        for batch_idx in range(batch_size):
            boxes = [
                torch.tensor(
                    [
                        x / width * self.original_img_size[1] - bbox_size / 2,
                        y / height * self.original_img_size[0] - bbox_size / 2,
                        x / width * self.original_img_size[1] + bbox_size / 2,
                        y / height * self.original_img_size[0] + bbox_size / 2,
                        1,
                        c,
                    ],
                    device=device,
                )
                for c in range(num_classes)
                for y, x in zip(
                    *torch.where(
                        torch.sigmoid(heatmap[batch_idx, c, :, :]) > 0.5
                    )
                )
            ]
            batch_boxes.append(
                torch.stack(boxes)
                if boxes
                else torch.empty((0, 6), device=device)
            )

        return batch_boxes
