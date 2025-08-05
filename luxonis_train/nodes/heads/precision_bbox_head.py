import math
from typing import Literal

import torch
from loguru import logger
from torch import Tensor, nn

from luxonis_train.nodes.blocks import DFL, ConvModule
from luxonis_train.nodes.heads import BaseHead
from luxonis_train.tasks import Tasks
from luxonis_train.typing import Packet
from luxonis_train.utils import (
    anchors_for_fpn_features,
    dist2bbox,
    non_max_suppression,
)


class PrecisionBBoxHead(BaseHead[list[Tensor], list[Tensor]]):
    in_channels: list[int]
    task = Tasks.BOUNDINGBOX
    parser = "YOLO"

    def __init__(
        self,
        reg_max: int = 16,
        n_heads: Literal[2, 3, 4] = 3,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        max_det: int = 300,
        **kwargs,
    ):
        """
        Adapted from U{Real-Time Flying Object Detection with YOLOv8
        <https://arxiv.org/pdf/2305.09972>} and from U{YOLOv6: A Single-Stage Object Detection Framework
        for Industrial Applications
        <https://arxiv.org/pdf/2209.02976.pdf>}.

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
        @type max_det: int
        @param max_det: Maximum number of detections retained after NMS.
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

        if len(self.in_channels) < self.n_heads:
            logger.warning(
                f"Head '{self.name}' was set to use {self.n_heads} heads, "
                f"but received only {len(self.in_channels)} inputs. "
                f"Changing number of heads to {len(self.in_channels)}."
            )
            self.n_heads = len(self.in_channels)

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
                    ConvModule(
                        x,
                        cls_channels,
                        kernel_size=3,
                        padding=1,
                        activation=nn.SiLU(),
                    ),
                    ConvModule(
                        cls_channels,
                        cls_channels,
                        kernel_size=3,
                        padding=1,
                        activation=nn.SiLU(),
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

        self.check_export_output_names()

    def check_export_output_names(self) -> None:
        if (
            self.export_output_names is None
            or len(self.export_output_names) != self.n_heads
        ):
            if (
                self.export_output_names is not None
                and len(self.export_output_names) != self.n_heads
            ):
                logger.warning(
                    f"Number of provided output names ({len(self.export_output_names)}) "
                    f"does not match number of heads ({self.n_heads}). "
                    f"Using default names."
                )
            self._export_output_names = [
                f"output{i + 1}_yolov8" for i in range(self.n_heads)
            ]

    def forward(self, x: list[Tensor]) -> tuple[list[Tensor], list[Tensor]]:
        cls_outputs = []
        reg_outputs = []
        for i in range(self.n_heads):
            reg_output = self.detection_heads[i][0](x[i])  # type: ignore
            cls_output = self.detection_heads[i][1](x[i])  # type: ignore
            reg_outputs.append(reg_output)
            cls_outputs.append(cls_output)
        return reg_outputs, cls_outputs

    def wrap(
        self, output: tuple[list[Tensor], list[Tensor]]
    ) -> Packet[Tensor]:
        reg_outputs, cls_outputs = (
            output  # ([bs, 4*reg_max, h_f, w_f]), ([bs, n_classes, h_f, w_f])
        )
        features = [
            torch.cat((reg, cls), dim=1)
            for reg, cls in zip(reg_outputs, cls_outputs)
        ]
        if self.training:
            return {
                "features": features,
            }

        if self.export:
            return {
                "boundingbox": self._prepare_bbox_export(
                    reg_outputs, cls_outputs
                )
            }

        boxes = non_max_suppression(
            self._prepare_bbox_inference_output(reg_outputs, cls_outputs),
            n_classes=self.n_classes,
            conf_thres=self.conf_thres,
            iou_thres=self.iou_thres,
            bbox_format="xyxy",
            max_det=self.max_det,
            predicts_objectness=False,
        )

        return {
            "features": features,
            "boundingbox": boxes,
        }

    def _fit_stride_to_n_heads(self) -> Tensor:
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

    def _prepare_bbox_and_cls(
        self, reg_outputs: list[Tensor], cls_outputs: list[Tensor]
    ) -> list[Tensor]:
        """Extract classification and bounding box tensors."""
        output = []
        for i in range(self.n_heads):
            box = self.dfl(reg_outputs[i])
            cls = cls_outputs[i].sigmoid()
            conf = cls.max(1, keepdim=True)[0]
            output.append(
                torch.cat([box, conf, cls], dim=1)
            )  # [bs, 4 + 1 + n_classes, h_f, w_f]
        return output

    def _prepare_bbox_export(
        self, reg_outputs: list[Tensor], cls_outputs: list[Tensor]
    ) -> list[Tensor]:
        """Prepare the output for export."""
        return self._prepare_bbox_and_cls(reg_outputs, cls_outputs)

    def _prepare_bbox_inference_output(
        self, reg_outputs: list[Tensor], cls_outputs: list[Tensor]
    ) -> Tensor:
        """Perform inference on predicted bounding boxes and class
        probabilities."""
        processed_outputs = self._prepare_bbox_and_cls(
            reg_outputs, cls_outputs
        )
        box_dists = []
        class_probs = []
        for feature in processed_outputs:
            bs, _, h, w = feature.size()
            reshaped = feature.view(bs, -1, h * w)
            box_dist = reshaped[:, :4, :]
            cls = reshaped[:, 5:, :]
            box_dists.append(box_dist)
            class_probs.append(cls)

        box_dists = torch.cat(box_dists, dim=2)
        class_probs = torch.cat(class_probs, dim=2)

        _, anchor_points, _, strides = anchors_for_fpn_features(
            processed_outputs, self.stride, 0.5
        )

        pred_bboxes = dist2bbox(
            box_dists, anchor_points.transpose(0, 1), out_format="xyxy", dim=1
        ) * strides.transpose(0, 1)

        base_output = [
            pred_bboxes.permute(0, 2, 1),  # [BS, H*W, 4]
            torch.ones(
                (box_dists.shape[0], pred_bboxes.shape[2], 1),
                dtype=pred_bboxes.dtype,
                device=pred_bboxes.device,
            ),
            class_probs.permute(0, 2, 1),  # [BS, H*W, n_classes]
        ]

        output_merged = torch.cat(
            base_output, dim=-1
        )  # [BS, H*W, 4 + 1 + n_classes]
        return output_merged

    def bias_init(self) -> None:
        """Initialize biases for the detection heads.

        Assumes detection_heads structure with separate regression and
        classification branches.
        """
        for head, stride in zip(self.detection_heads, self.stride):
            reg_branch = head[0]  # type: ignore
            cls_branch = head[1]  # type: ignore

            reg_conv = reg_branch[-1]
            reg_conv.bias.data[:] = 1.0

            cls_conv = cls_branch[-1]
            cls_conv.bias.data[: self.n_classes] = math.log(
                5 / self.n_classes / (self.original_in_shape[1] / stride) ** 2
            )

    def initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                pass
            elif isinstance(m, nn.BatchNorm2d):
                m.eps = 0.001
                m.momentum = 0.03
            elif isinstance(
                m, nn.Hardswish | nn.LeakyReLU | nn.ReLU | nn.ReLU6 | nn.SiLU
            ):
                m.inplace = True

    def get_custom_head_config(self) -> dict:
        """Returns custom head configuration.

        @rtype: dict
        @return: Custom head configuration.
        """
        return {
            "subtype": "yolov8",
            "iou_threshold": self.iou_thres,
            "conf_threshold": self.conf_thres,
            "max_det": self.max_det,
        }
