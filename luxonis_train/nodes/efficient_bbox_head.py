"""Head for object detection.

Adapted from U{YOLOv6: A Single-Stage Object Detection Framework for Industrial
Applications<https://arxiv.org/pdf/2209.02976.pdf>}.
"""

from typing import Literal

import torch
from torch import Tensor, nn

from luxonis_train.nodes.blocks import EfficientDecoupledBlock
from luxonis_train.utils.boxutils import (
    anchors_for_fpn_features,
    dist2bbox,
    non_max_suppression,
)
from luxonis_train.utils.types import LabelType, Packet

from .base_node import BaseNode


class EfficientBBoxHead(
    BaseNode[list[Tensor], tuple[list[Tensor], list[Tensor], list[Tensor]]]
):
    in_channels: list[int]

    def __init__(
        self,
        n_heads: Literal[2, 3, 4] = 3,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        **kwargs,
    ):
        """Head for object detection.

        TODO: add more documentation

        @type n_heads: Literal[2,3,4]
        @param n_heads: Number of output heads. Defaults to 3.
          ***Note:*** Should be same also on neck in most cases.

        @type conf_thres: float
        @param conf_thres: Threshold for confidence. Defaults to C{0.25}.

        @type iou_thres: float
        @param iou_thres: Threshold for IoU. Defaults to C{0.45}.
        """
        super().__init__(task_type=LabelType.BOUNDINGBOX, **kwargs)

        self.n_heads = n_heads

        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        self.stride = self._fit_stride_to_num_heads()
        self.grid_cell_offset = 0.5
        self.grid_cell_size = 5.0

        self.heads = nn.ModuleList()
        for i in range(self.n_heads):
            curr_head = EfficientDecoupledBlock(
                n_classes=self.n_classes,
                in_channels=self.in_channels[i],
            )
            self.heads.append(curr_head)

    def forward(
        self, inputs: list[Tensor]
    ) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
        features: list[Tensor] = []
        cls_score_list: list[Tensor] = []
        reg_distri_list: list[Tensor] = []

        for i, module in enumerate(self.heads):
            out_feature, out_cls, out_reg = module(inputs[i])
            features.append(out_feature)
            out_cls = torch.sigmoid(out_cls)
            cls_score_list.append(out_cls)
            reg_distri_list.append(out_reg)

        return features, cls_score_list, reg_distri_list

    def wrap(
        self, output: tuple[list[Tensor], list[Tensor], list[Tensor]]
    ) -> Packet[Tensor]:
        features, cls_score_list, reg_distri_list = output

        if self.export:
            outputs = []
            for out_cls, out_reg in zip(cls_score_list, reg_distri_list, strict=True):
                conf, _ = out_cls.max(1, keepdim=True)
                out = torch.cat([out_reg, conf, out_cls], dim=1)
                outputs.append(out)
            return {"boxes": outputs}

        cls_tensor = torch.cat(
            [cls_score_list[i].flatten(2) for i in range(len(cls_score_list))], dim=2
        ).permute(0, 2, 1)
        reg_tensor = torch.cat(
            [reg_distri_list[i].flatten(2) for i in range(len(reg_distri_list))], dim=2
        ).permute(0, 2, 1)

        if self.training:
            return {
                "features": features,
                "class_scores": [cls_tensor],
                "distributions": [reg_tensor],
            }

        else:
            boxes = self._process_to_bbox((features, cls_tensor, reg_tensor))
            return {
                "boxes": boxes,
                "features": features,
                "class_scores": [cls_tensor],
                "distributions": [reg_tensor],
            }

    def _fit_stride_to_num_heads(self):
        """Returns correct stride for number of heads and attach index."""
        stride = torch.tensor(
            [
                self.original_in_shape[2] / x[2]  # type: ignore
                for x in self.in_sizes[: self.n_heads]
            ],
            dtype=torch.int,
        )
        return stride

    def _process_to_bbox(
        self, output: tuple[list[Tensor], Tensor, Tensor]
    ) -> list[Tensor]:
        """Performs post-processing of the output and returns bboxs after NMS."""
        features, cls_score_list, reg_dist_list = output
        _, anchor_points, _, stride_tensor = anchors_for_fpn_features(
            features,
            self.stride,
            self.grid_cell_size,
            self.grid_cell_offset,
            multiply_with_stride=False,
        )

        pred_bboxes = dist2bbox(reg_dist_list, anchor_points, out_format="xyxy")

        pred_bboxes *= stride_tensor
        output_merged = torch.cat(
            [
                pred_bboxes,
                torch.ones(
                    (features[-1].shape[0], pred_bboxes.shape[1], 1),
                    dtype=pred_bboxes.dtype,
                    device=pred_bboxes.device,
                ),
                cls_score_list,
            ],
            dim=-1,
        )

        return non_max_suppression(
            output_merged,
            n_classes=self.n_classes,
            conf_thres=self.conf_thres,
            iou_thres=self.iou_thres,
            bbox_format="xyxy",
            predicts_objectness=False,
        )
