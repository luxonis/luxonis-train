import math
from typing import Literal

import torch
from torch import Tensor, nn

from luxonis_train.nodes.blocks import EfficientOBBDecoupledBlock
from luxonis_train.nodes.heads import EfficientBBoxHead
from luxonis_train.utils.boxutils import (
    anchors_for_fpn_features,
    dist2rbbox,
    non_max_suppression_obb,
)
from luxonis_train.utils.types import LabelType, Packet


class EfficientOBBoxHead(EfficientBBoxHead):
    tasks: list[LabelType] = [LabelType.OBOUNDINGBOX]

    def __init__(
        self,
        n_heads: Literal[2, 3, 4] = 3,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        max_det: int = 300,
        reg_max: int = 16,
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

        @type max_det: int
        @param max_det: Maximum number of detections retained after NMS. Defaults to C{300}.

        @type reg_max: int
        @param reg_max: Number of bins for predicting the distributions of bounding box coordinates.
        """
        super().__init__(n_heads, conf_thres, iou_thres, max_det, **kwargs)

        self.reg_max = reg_max

        self.heads = nn.ModuleList()
        for i in range(self.n_heads):
            curr_head = EfficientOBBDecoupledBlock(
                n_classes=self.n_classes,
                in_channels=self.in_channels[i],
            )
            self.heads.append(curr_head)

    def forward(
        self, inputs: list[Tensor]
    ) -> tuple[list[Tensor], list[Tensor], list[Tensor], list[Tensor]]:
        features: list[Tensor] = []
        cls_score_list: list[Tensor] = []
        reg_distri_list: list[Tensor] = []
        angles_list: list[Tensor] = []

        for i, module in enumerate(self.heads):
            out_feature, out_cls, out_reg, out_angle = module(inputs[i])
            features.append(out_feature)

            out_cls = torch.sigmoid(out_cls)
            cls_score_list.append(out_cls)

            reg_distri_list.append(out_reg)

            # out_angle = (out_angle.sigmoid() - 0.25) * math.pi  # [-pi/4, 3pi/4]
            out_angle = out_angle.sigmoid() * math.pi / 2  # [0, pi/2]
            angles_list.append(out_angle)

        return features, cls_score_list, reg_distri_list, angles_list

    def wrap(
        self, output: tuple[list[Tensor], list[Tensor], list[Tensor], list[Tensor]]
    ) -> Packet[Tensor]:
        features, cls_score_list, reg_distri_list, angles_list = output

        if self.export:
            outputs = []
            for out_cls, out_reg, out_angles in zip(
                cls_score_list, reg_distri_list, angles_list, strict=True
            ):
                conf, _ = out_cls.max(1, keepdim=True)
                out = torch.cat([out_reg, conf, out_cls, out_angles], dim=1)
                outputs.append(out)
            return {self.task: outputs}

        angle_tensor = torch.cat(
            [angles_list[i].flatten(2) for i in range(len(angles_list))], dim=2
        ).permute(0, 2, 1)
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
                "angles": [angle_tensor],
            }

        else:
            boxes = self._process_to_bbox(
                (features, cls_tensor, reg_tensor, angle_tensor)
            )
            return {
                "oboundingbox": boxes,
                "features": features,
                "class_scores": [cls_tensor],
                "distributions": [reg_tensor],
                "angles": [angle_tensor],
            }

    def _process_to_bbox(
        self, output: tuple[list[Tensor], Tensor, Tensor, Tensor]
    ) -> list[Tensor]:
        """Performs post-processing of the output and returns bboxs after NMS."""
        features, cls_score_tensor, reg_dist_tensor, angles_tensor = output
        _, anchor_points, _, stride_tensor = anchors_for_fpn_features(
            features,
            self.stride,
            self.grid_cell_size,
            self.grid_cell_offset,
            multiply_with_stride=False,
        )

        # The following block below is implied for the distributed predictions of the regression
        # branch (used in DFL)
        # if self.use_dfl: # consider adding this as a parameter
        proj = torch.arange(
            self.reg_max, dtype=torch.float, device=reg_dist_tensor.device
        )
        b, a, c = reg_dist_tensor.shape  # batch, anchors, channels
        reg_dist_mean_tensor = (  # we get a tensor of the expected values (mean) of the regression predictions
            reg_dist_tensor.view(b, a, 4, c // 4)
            .softmax(3)
            .matmul(proj.type(reg_dist_tensor.dtype))
        )
        pred_bboxes = torch.cat(
            (
                dist2rbbox(reg_dist_mean_tensor, angles_tensor, anchor_points),
                angles_tensor,
            ),
            dim=-1,
        )  # xywhr

        xy_strided = pred_bboxes[..., :2] * stride_tensor
        pred_bboxes = torch.cat(
            [xy_strided, pred_bboxes[..., 2:]], dim=-1
        )  # xywhr with xy strided

        output_merged = torch.cat(
            [
                pred_bboxes,
                torch.ones(
                    (features[-1].shape[0], pred_bboxes.shape[1], 1),
                    dtype=pred_bboxes.dtype,
                    device=pred_bboxes.device,
                ),
                cls_score_tensor,
            ],
            dim=-1,
        )

        # pred = torch.rand((2, 1344, 15), device=pred_bboxes.device)
        # pred[..., 5] = 1

        return non_max_suppression_obb(
            output_merged,
            # pred,  # for debugging
            n_classes=self.n_classes,
            conf_thres=self.conf_thres,
            iou_thres=self.iou_thres,
            max_det=self.max_det,
            predicts_objectness=False,
        )
