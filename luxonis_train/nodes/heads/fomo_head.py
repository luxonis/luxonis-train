import torch
import torch.nn.functional as F
from torch import Tensor, nn
from typing_extensions import override

from luxonis_train.nodes.base_node import BaseNode
from luxonis_train.tasks import Task, Tasks
from luxonis_train.utils import Packet


class FOMOHead(BaseNode[list[Tensor], list[Tensor]]):
    task: Task = Tasks.FOMO
    in_channels: int
    attach_index: int = 1

    def __init__(
        self,
        num_conv_layers: int = 3,
        conv_channels: int = 16,
        use_nms: bool = True,
        **kwargs,
    ):
        """FOMO Head for object detection using heatmaps.

        @type num_conv_layers: int
        @param num_conv_layers: Number of convolutional layers to use.
        @type conv_channels: int
        @param conv_channels: Number of channels to use in the
            convolutional layers.
        """
        super().__init__(**kwargs)
        self.original_img_size = self.original_in_shape[1:]
        self.num_conv_layers = num_conv_layers
        self.conv_channels = conv_channels
        self.use_nms = use_nms

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

    @property
    @override
    def n_keypoints(self) -> int:
        return 1

    def forward(self, inputs: list[Tensor]) -> Tensor:
        return self.conv_layers(inputs)

    def wrap(self, heatmap: Tensor) -> Packet[Tensor]:
        if self.training:
            return {self.task.main_output: heatmap}

        if self.export:
            return {"outputs": [self._apply_nms_if_needed(heatmap)]}

        keypoints = self._heatmap_to_kpts(heatmap)
        return {
            "keypoints": keypoints,  # format is list of tensors of shape [N,nk,4] where 4 because of x,y,v, cls_id
            self.task.main_output: heatmap,
            "boundingbox": self._keypoints_to_bboxes(  # Bboxes are in list of tesnors of shape [N,6] where 6 because of cls_id, score, x,y,x,y,score, cls_id
                keypoints, self.original_img_size[0], self.original_img_size[1]
            ),
        }

    def _apply_nms_if_needed(self, heatmap: Tensor) -> Tensor:
        """Apply NMS pooling to the heatmap if use_nms is enabled.

        @type heatmap: Tensor
        @param heatmap: Heatmap to process.
        @return: Processed heatmap with or without pooling.
        """
        if not self.use_nms:
            return heatmap

        return F.max_pool2d(heatmap, kernel_size=3, stride=1, padding=1)

    def _heatmap_to_kpts(self, heatmap: Tensor) -> list[Tensor]:
        """Convert heatmap to keypoint pairs using local-max NMS so that
        only the strongest local peak in a neighborhood is retained.

        @type heatmap: Tensor
        @param heatmap: Heatmap to convert to keypoints.
        """
        device = heatmap.device
        batch_size, num_classes, height, width = heatmap.shape

        batch_kpts = []
        for batch_idx in range(batch_size):
            kpts_per_img = []

            for class_id in range(num_classes):
                prob_map = torch.sigmoid(heatmap[batch_idx, class_id, :, :])

                keep = self._get_keypoint_mask(prob_map)

                y_indices, x_indices = torch.where(keep)
                kpts = [
                    [
                        x.item() / width * self.original_img_size[1],
                        y.item() / height * self.original_img_size[0],
                        float(prob_map[y, x]),
                        class_id,
                    ]
                    for y, x in zip(y_indices, x_indices)
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
                    prob_map.unsqueeze(0).unsqueeze(0),  # [1, 1, H, W]
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

    @staticmethod
    def _keypoints_to_bboxes(
        keypoints: list[Tensor],
        img_height: int,
        img_width: int,
        box_width: int = 2,
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
            (cls_id, score, x_min, y_min, x_max, y_max).
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

            x_min = (x_coords - half_box).clamp(min=0) / img_width
            y_min = (y_coords - half_box).clamp(min=0) / img_height
            x_max = (x_coords + half_box).clamp(max=img_width) / img_width
            y_max = (y_coords + half_box).clamp(max=img_height) / img_height
            bboxes = torch.stack(
                [cls_ids, scores, x_min, y_min, x_max, y_max], dim=-1
            )
            bboxes_list.append(bboxes)

        return bboxes_list
