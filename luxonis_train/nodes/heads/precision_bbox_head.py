import math
from typing import Literal, cast

import torch
from luxonis_ml.typing import Params
from torch import Tensor, nn
from typing_extensions import override

from luxonis_train.nodes.blocks import DFL
from luxonis_train.nodes.blocks.blocks import PreciseDecoupledBlock
from luxonis_train.tasks import Tasks
from luxonis_train.typing import Packet
from luxonis_train.utils import (
    anchors_for_fpn_features,
    dist2bbox,
    non_max_suppression,
)

from .base_detection_head import BaseDetectionHead


class PrecisionBBoxHead(BaseDetectionHead):
    task = Tasks.BOUNDINGBOX

    def __init__(
        self,
        n_heads: Literal[2, 3, 4] = 3,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        max_det: int = 300,
        reg_max: int = 16,
        **kwargs,
    ):
        """
        Adapted from U{Real-Time Flying Object Detection with YOLOv8
        <https://arxiv.org/pdf/2305.09972>} and from U{YOLOv6: A Single-Stage Object Detection Framework
        for Industrial Applications
        <https://arxiv.org/pdf/2209.02976.pdf>}.

        @type n_heads: Literal[2, 3, 4]
        @param n_heads: Number of output heads.
        @type conf_thres: float
        @param conf_thres: Confidence threshold for NMS.
        @type iou_thres: float
        @param iou_thres: IoU threshold for NMS.
        @type max_det: int
        @param max_det: Maximum number of detections retained after NMS.
        @type reg_max: int
        @param reg_max: Maximum number of regression channels.
        """
        super().__init__(
            n_heads=n_heads,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            max_det=max_det,
            **kwargs,
        )
        self.reg_max = reg_max
        self.no = self.n_classes + reg_max * 4
        self.grid_cell_offset = 0.5
        self.grid_cell_size = 5.0

        reg_channels = max((16, self.in_channels[0] // 4, reg_max * 4))
        cls_channels = max(self.in_channels[0], min(self.n_classes, 100))

        self.heads = cast(
            list[PreciseDecoupledBlock],
            nn.ModuleList(
                PreciseDecoupledBlock(
                    in_channels=in_channels,
                    reg_channels=reg_channels,
                    cls_channels=cls_channels,
                    n_classes=self.n_classes,
                    reg_max=reg_max,
                )
                for in_channels in self.in_channels
            ),
        )

        self.dfl = DFL(reg_max) if reg_max > 1 else nn.Identity()

    def forward(self, inputs: list[Tensor]) -> Packet[Tensor]:
        features_list = []
        classes_list = []
        regressions_list = []
        for head, x in zip(self.heads, inputs, strict=True):
            features, classes, regressions = head(x)
            regressions_list.append(regressions)
            classes_list.append(classes)
            features_list.append(features)

        if self.training:
            return {"features": features_list}

        if self.export:
            return {
                "boundingbox": self._construct_raw_bboxes(
                    classes_list, regressions_list
                )
            }

        boxes = non_max_suppression(
            self._prepare_bbox_inference_output(
                classes_list, regressions_list
            ),
            n_classes=self.n_classes,
            conf_thres=self.conf_thres,
            iou_thres=self.iou_thres,
            bbox_format="xyxy",
            max_det=self.max_det,
            predicts_objectness=False,
        )

        return {
            "features": features_list,
            "boundingbox": boxes,
        }

    @override
    def initialize_weights(self, method: str | None = None) -> None:
        """Initialize biases for the detection heads.

        Assumes detection_heads structure with separate regression and
        classification branches.
        """
        super().initialize_weights(method)
        for head, stride in zip(self.heads, self.stride, strict=True):
            reg_conv = head.regression_branch[-1]
            assert isinstance(reg_conv, nn.Conv2d)
            if reg_conv.bias is not None:
                nn.init.constant_(reg_conv.bias, 1.0)

            cls_conv = head.classification_branch[-1]
            assert isinstance(cls_conv, nn.Conv2d)
            if cls_conv.bias is not None:
                cls_conv.bias.data[: self.n_classes] = math.log(
                    5
                    / self.n_classes
                    / (self.original_in_shape[1] / stride) ** 2
                )

    @property
    @override
    def export_output_names(self) -> list[str] | None:
        return self.get_output_names(
            [f"output{i + 1}_yolov8" for i in range(self.n_heads)]
        )

    @override
    def get_custom_head_config(self) -> Params:
        """Returns custom head configuration.

        @rtype: dict
        @return: Custom head configuration.
        """
        return super().get_custom_head_config() | {"subtype": "yolov8"}

    def _construct_raw_bboxes(
        self, classes_list: list[Tensor], regressions_list: list[Tensor]
    ) -> list[Tensor]:
        """Extract classification and bounding box tensors."""
        bboxes = []
        for i in range(self.n_heads):
            bbox = self.dfl(regressions_list[i])
            classes = classes_list[i].sigmoid()
            confidence = classes.max(1, keepdim=True)[0]
            # @shape: [N, 4 + 1 + n_classes, h_f, w_f]
            bboxes.append(torch.cat([bbox, confidence, classes], dim=1))
        return bboxes

    def _prepare_bbox_inference_output(
        self, classes_list: list[Tensor], regressions_list: list[Tensor]
    ) -> Tensor:
        """Perform inference on predicted bounding boxes and class
        probabilities."""
        raw_bboxes = self._construct_raw_bboxes(classes_list, regressions_list)
        bbox_distributions = []
        class_probabilities = []
        for raw_bbox in raw_bboxes:
            bs, _, h, w = raw_bbox.size()
            raw_bbox = raw_bbox.view(bs, -1, h * w)
            confidence = raw_bbox[:, :4, :]
            classes = raw_bbox[:, 5:, :]
            bbox_distributions.append(confidence)
            class_probabilities.append(classes)

        bbox_distributions = torch.cat(bbox_distributions, dim=2)
        class_probabilities = torch.cat(class_probabilities, dim=2)

        _, anchor_points, _, strides = anchors_for_fpn_features(
            raw_bboxes, self.stride, 0.5
        )

        pred_bboxes = dist2bbox(
            bbox_distributions,
            anchor_points.transpose(0, 1),
            out_format="xyxy",
            dim=1,
        ) * strides.transpose(0, 1)

        base_output = [
            # @shape: [N, H * W, 4]
            pred_bboxes.permute(0, 2, 1),
            torch.ones(
                (bbox_distributions.shape[0], pred_bboxes.shape[2], 1),
                dtype=pred_bboxes.dtype,
                device=pred_bboxes.device,
            ),
            # @shape: [N, H * W, n_classes]
            class_probabilities.permute(0, 2, 1),
        ]

        # @shape: [N, H * W, 4 + 1 + n_classes]
        return torch.cat(base_output, dim=-1)
