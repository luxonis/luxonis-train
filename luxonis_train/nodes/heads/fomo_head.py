import torch
import torch.nn.functional as F
from torch import Tensor, nn
from typing_extensions import override

from luxonis_train.nodes.blocks.blocks import ConvBlock
from luxonis_train.nodes.heads import BaseHead
from luxonis_train.tasks import Task, Tasks
from luxonis_train.typing import Packet


class FOMOHead(BaseHead):
    task: Task = Tasks.FOMO
    in_channels: int

    def __init__(
        self,
        n_conv_layers: int = 3,
        conv_channels: int = 16,
        use_nms: bool = True,
        **kwargs,
    ):
        """FOMO Head for object detection using heatmaps.

        @type n_conv_layers: int
        @param n_conv_layers: Number of convolutional layers to use.
        @type conv_channels: int
        @param conv_channels: Number of channels to use in the
            convolutional layers.
        """
        super().__init__(**kwargs)
        self.n_conv_layers = n_conv_layers
        self.conv_channels = conv_channels
        self.use_nms = use_nms

        current_channels = self.in_channels

        layers = []
        for _ in range(self.n_conv_layers - 1):
            layers.append(
                ConvBlock(
                    current_channels,
                    self.conv_channels,
                    kernel_size=1,
                    stride=1,
                    use_norm=False,
                    bias=True,
                )
            )
            current_channels = self.conv_channels
        layers.append(
            nn.Conv2d(
                self.conv_channels, self.n_classes, kernel_size=1, stride=1
            )
        )
        self.conv_layers = nn.Sequential(*layers)

    @property
    @override
    def n_keypoints(self) -> int:
        return 1

    def forward(self, inputs: Tensor) -> Packet[Tensor]:
        heatmap = self.conv_layers(inputs)

        if self.training:
            return {self.task.main_output: heatmap}

        if self.export:
            if self.use_nms:
                heatmap = F.max_pool2d(
                    heatmap, kernel_size=3, stride=1, padding=1
                )

            return {"outputs": [heatmap]}

        return {
            "keypoints": self._heatmap_to_kpts(heatmap),
            self.task.main_output: heatmap,
        }

    def _heatmap_to_kpts(self, heatmap: Tensor) -> list[Tensor]:
        """Convert heatmap to keypoint pairs using local-max NMS so that
        only the strongest local peak in a neighborhood is retained.

        @type heatmap: Tensor
        @param heatmap: Heatmap to convert to keypoints.
        """
        device = heatmap.device
        batch_size, n_classes, height, width = heatmap.shape

        batch_kpts = []
        for batch_idx in range(batch_size):
            kpts_per_img = []

            for class_id in range(n_classes):
                prob_map = torch.sigmoid(heatmap[batch_idx, class_id, :, :])

                keep = self._get_keypoint_mask(prob_map)

                y_indices, x_indices = torch.where(keep)
                kpts = [
                    [
                        x.item() / width * self.original_in_shape[2],
                        y.item() / height * self.original_in_shape[1],
                        float(prob_map[y, x]),
                        class_id,
                    ]
                    for y, x in zip(y_indices, x_indices, strict=True)
                ]

                kpts_per_img.extend(kpts)

            batch_kpt = torch.tensor(
                kpts_per_img, device=device, dtype=torch.float32
            ).unsqueeze(1)
            if batch_kpt.numel() == 0:
                batch_kpt = torch.zeros((0, 1, 4), device=device)
            batch_kpts.append(batch_kpt)

        return batch_kpts

    def _get_keypoint_mask(self, prob_map: Tensor) -> Tensor:
        """Generate a mask for keypoints using NMS if enabled.

        @type prob_map: Tensor
        @param prob_map: Probability map for a specific class.
        @return: Binary mask indicating keypoint positions.
        """
        if self.use_nms:
            pooled_map = (
                F.max_pool2d(
                    # Shape: `[1, 1, H, W]`
                    prob_map.unsqueeze(0).unsqueeze(0),
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
                .squeeze(0)
                .squeeze(0)
            )  # [H, W]
            threshold = 0.5
            return (prob_map == pooled_map) & (prob_map > threshold)

        return prob_map > 0.5
