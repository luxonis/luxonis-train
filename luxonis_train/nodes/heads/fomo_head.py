import logging
from typing import Any, List

import torch
from torch import Tensor, nn

from luxonis_train.enums import TaskType
from luxonis_train.nodes.base_node import BaseNode
from luxonis_train.utils import Packet

logger = logging.getLogger(__name__)


class FOMOHead(BaseNode[list[Tensor], list[Tensor]]):
    tasks: list[TaskType] = [TaskType.BOUNDINGBOX]
    in_channels: int
    attach_index: int = 2

    def __init__(
        self,
        num_conv_layers: int = 3,
        conv_channels: int = 16,
        **kwargs: Any,
    ):
        """FOMO Head for object detection using heatmaps.

        @type num_conv_layers: int
        @param num_conv_layers: Number of convolutional layers to use.
        @type conv_channels: int
        @param conv_channels: Number of channels to use in the
            convolutional layers.
        @type kwargs: Any
        @param kwargs: Additional arguments.
        """
        super().__init__(**kwargs)
        self.original_img_size = self.original_in_shape[1:]
        self.num_conv_layers = num_conv_layers
        self.conv_channels = conv_channels

        current_channels = self.in_channels

        layers = []
        for _ in range(self.num_conv_layers - 1):
            layers.append(
                nn.Conv2d(
                    current_channels,
                    self.conv_channels,
                    kernel_size=1,
                    stride=1,
                )
            )
            layers.append(nn.ReLU())
            current_channels = self.conv_channels
        layers.append(
            nn.Conv2d(
                self.conv_channels, self.n_classes, kernel_size=1, stride=1
            )
        )
        self.conv_layers = nn.Sequential(*layers)

    def forward(self, inputs: List[Tensor]) -> Tensor:
        return self.conv_layers(inputs)

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
