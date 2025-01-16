import logging
from typing import Any, List

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from luxonis_train.enums import TaskType
from luxonis_train.nodes.base_node import BaseNode
from luxonis_train.utils import Packet

logger = logging.getLogger(__name__)


class FOMOHead(BaseNode[list[Tensor], list[Tensor]]):
    tasks: list[TaskType] = [TaskType.KEYPOINTS, TaskType.BOUNDINGBOX]
    in_channels: int
    attach_index: int = 1

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
            keypoints = self._heatmap_to_kpts(heatmap)
            return {
                "keypoints": keypoints,
                "features": [heatmap],
            }

    def _heatmap_to_kpts(self, heatmap: Tensor) -> List[Tensor]:
        """Convert heatmap to keypoint pairs using local-max NMS so that
        only the strongest local peak in a neighborhood is retained."""
        device = heatmap.device
        batch_size, num_classes, height, width = heatmap.shape

        batch_kpts = []
        for batch_idx in range(batch_size):
            kpts_per_img = []

            for c in range(num_classes):
                prob_map = torch.sigmoid(heatmap[batch_idx, c, :, :])

                pooled_map = (
                    F.max_pool2d(
                        prob_map.unsqueeze(0).unsqueeze(0),  # shape [1,1,H,W]
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    )
                    .squeeze(0)
                    .squeeze(0)
                )  # back to [H,W]

                threshold = 0.5
                keep = (prob_map == pooled_map) & (prob_map > threshold)

                y_indices, x_indices = torch.where(keep)
                kpts = []
                for y, x in zip(y_indices, x_indices):
                    kpt_x = x.item() / width * self.original_img_size[1]
                    kpt_y = y.item() / height * self.original_img_size[0]

                    kpts.append([kpt_x, kpt_y, float(prob_map[y, x])])

                kpts_per_img.append(kpts)

            if all(len(kpt) == 0 for kpt in kpts_per_img):
                kpts_per_img = [[[0, 0, 0.0]]]

            batch_kpts.append(
                torch.tensor(
                    kpts_per_img, device=device, dtype=torch.float32
                ).permute(1, 0, 2)
            )

        return batch_kpts
