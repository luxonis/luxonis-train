import logging
import math
from typing import Any, Literal

import torch
from torch import Tensor, nn

from luxonis_train.enums import TaskType
from luxonis_train.nodes import BaseNode
from luxonis_train.nodes.blocks import DFL, ConvModule, DWConvModule
from luxonis_train.utils import (
    Packet,
    anchors_for_fpn_features,
    dist2bbox,
    non_max_suppression,
)

logger = logging.getLogger(__name__)


class PrecisionBBoxHead(BaseNode[list[Tensor], list[Tensor]]):
    in_channels: list[int]
    tasks: list[TaskType] = [TaskType.BOUNDINGBOX]

    def __init__(
        self,
        reg_max: int = 16,
        n_heads: Literal[2, 3, 4] = 3,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        max_det: int = 300,
        **kwargs: Any,
    ):
        """
        Adapted from U{Real-Time Flying Object Detection with YOLOv8
        <https://arxiv.org/pdf/2305.09972>}

        @type ch: tuple[int]
        @param ch: Channels for each detection layer.
        @type reg_max: int
        @param reg_max: Maximum number of regression channels.
        @type n_heads: Literal[2, 3, 4]
        @param n_heads: Number of output heads.
        @type conf_thres: float
        @param conf_thres: Confidence threshold for NMS.
        @type iou_thres: float
        @param iou_thres: IoU threshold for NMS.
        """
        super().__init__(**kwargs)
        self.reg_max = reg_max
        self.no = self.n_classes + reg_max * 4
        self.n_heads = n_heads
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.grid_cell_offset = 0.5
        self.grid_cell_size = 5.0
        self.max_det = max_det

        reg_channels = max((16, self.in_channels[0] // 4, reg_max * 4))
        cls_channels = max(self.in_channels[0], min(self.n_classes, 100))

        self.detection_heads = nn.ModuleList(
            nn.Sequential(
                # Regression branch
                nn.Sequential(
                    ConvModule(
                        x,
                        reg_channels,
                        kernel_size=3,
                        padding=1,
                        activation=nn.SiLU(),
                    ),
                    ConvModule(
                        reg_channels,
                        reg_channels,
                        kernel_size=3,
                        padding=1,
                        activation=nn.SiLU(),
                    ),
                    nn.Conv2d(reg_channels, 4 * self.reg_max, kernel_size=1),
                ),
                # Classification branch
                nn.Sequential(
                    nn.Sequential(
                        DWConvModule(
                            x,
                            x,
                            kernel_size=3,
                            padding=1,
                            activation=nn.SiLU(),
                        ),
                        ConvModule(
                            x,
                            cls_channels,
                            kernel_size=1,
                            activation=nn.SiLU(),
                        ),
                    ),
                    nn.Sequential(
                        DWConvModule(
                            cls_channels,
                            cls_channels,
                            kernel_size=3,
                            padding=1,
                            activation=nn.SiLU(),
                        ),
                        ConvModule(
                            cls_channels,
                            cls_channels,
                            kernel_size=1,
                            activation=nn.SiLU(),
                        ),
                    ),
                    nn.Conv2d(cls_channels, self.n_classes, kernel_size=1),
                ),
            )
            for x in self.in_channels
        )

        self.stride = self._fit_stride_to_n_heads()
        self.dfl = DFL(reg_max) if reg_max > 1 else nn.Identity()
        self.bias_init()
        self.initialize_weights()

    def forward(self, x: list[Tensor]) -> list[Tensor]:
        for i in range(self.n_heads):
            reg_output = self.detection_heads[i][0](x[i])
            cls_output = self.detection_heads[i][1](x[i])
            x[i] = torch.cat((reg_output, cls_output), 1)
        return x

    def wrap(self, output: list[Tensor]) -> Packet[Tensor]:
        if self.training:
            return {
                "features": output,
            }
        y = self._inference(output)
        if self.export:
            return {self.task: y}
        boxes = non_max_suppression(
            y,
            n_classes=self.n_classes,
            conf_thres=self.conf_thres,
            iou_thres=self.iou_thres,
            bbox_format="xyxy",
            max_det=self.max_det,
            predicts_objectness=False,
        )

        return {
            "features": output,
            "boundingbox": boxes,
        }

    def _fit_stride_to_n_heads(self):
        """Returns correct stride for number of heads and attach
        index."""
        stride = torch.tensor(
            [
                self.original_in_shape[1] / x[2]  # type: ignore
                for x in self.in_sizes[: self.n_heads]
            ],
            dtype=torch.int,
        )
        return stride

    def _inference(self, x: list[Tensor], masks: Tensor | None = None):
        """Decode predicted bounding boxes and class probabilities based
        on multiple-level feature maps."""
        shape = x[0].shape
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        _, self.anchor_points, _, self.strides = anchors_for_fpn_features(
            x, self.stride, 0.5
        )
        box, cls = x_cat.split((self.reg_max * 4, self.n_classes), 1)
        pred_bboxes = self.decode_bboxes(
            self.dfl(box), self.anchor_points.transpose(0, 1)
        ) * self.strides.transpose(0, 1)

        if self.export:
            return torch.cat(
                (pred_bboxes.permute(0, 2, 1), cls.sigmoid().permute(0, 2, 1)),
                1,
            )

        base_output = [
            pred_bboxes.permute(0, 2, 1),
            torch.ones(
                (shape[0], pred_bboxes.shape[2], 1),
                dtype=pred_bboxes.dtype,
                device=pred_bboxes.device,
            ),
            cls.permute(0, 2, 1),
        ]

        if masks is not None:
            base_output.append(masks.permute(0, 2, 1))

        output_merged = torch.cat(base_output, dim=-1)
        return output_merged

    def decode_bboxes(self, bboxes: Tensor, anchors: Tensor) -> Tensor:
        """Decode bounding boxes."""
        return dist2bbox(bboxes, anchors, out_format="xyxy", dim=1)

    def bias_init(self):
        """Initialize biases for the detection heads.

        Assumes detection_heads structure with separate regression and
        classification branches.
        """
        for head, stride in zip(self.detection_heads, self.stride):
            reg_branch = head[0]
            cls_branch = head[1]

            reg_conv = reg_branch[-1]
            reg_conv.bias.data[:] = 1.0

            cls_conv = cls_branch[-1]
            cls_conv.bias.data[: self.n_classes] = math.log(
                5 / self.n_classes / (self.original_in_shape[1] / stride) ** 2
            )

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                pass
            elif isinstance(m, nn.BatchNorm2d):
                m.eps = 0.001
                m.momentum = 0.03
            elif isinstance(
                m, (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU)
            ):
                m.inplace = True
