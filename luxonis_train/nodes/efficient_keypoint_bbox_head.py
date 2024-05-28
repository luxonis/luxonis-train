from typing import Literal
import torch
from torch import Tensor, nn
import math

from luxonis_train.nodes.blocks import ConvModule

from luxonis_train.utils.types import LabelType, Packet
from luxonis_train.utils.boxutils import (
    anchors_for_fpn_features,
    dist2bbox,
    non_max_suppression,
)
from .efficient_bbox_head import EfficientBBoxHead 

class EfficientKeypointBBoxHead(EfficientBBoxHead):
    def __init__(
        self,
        n_keypoints: int | None = None,
        n_heads: Literal[2, 3, 4] = 3,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        max_det: int = 300,
        **kwargs,
    ):
        """Head for object detection and keypoint detection."""
        super().__init__(n_heads=n_heads, conf_thres=conf_thres, iou_thres=iou_thres, max_det=max_det, **kwargs)

        n_keypoints = n_keypoints or self.dataset_metadata._n_keypoints
        self.n_keypoints = n_keypoints
        self.kpt_shape = (n_keypoints, 3)
        self.nk = n_keypoints * 3

        c4 = max(self.in_channels[0] // 4, self.nk)
        self.kpt_layers = nn.ModuleList(
            nn.Sequential(
                ConvModule(x, c4, 3, 1, 1, activation= nn.SiLU()), 
                ConvModule(c4, c4, 3, 1, 1, activation= nn.SiLU()),
                nn.Conv2d(c4, self.nk, 1, 1)
            ) for x in self.in_channels
        )

    def forward(self, inputs: list[Tensor]) -> tuple[list[Tensor], list[Tensor], list[Tensor], list[Tensor]]:
        features, cls_score_list, reg_distri_list = super().forward(inputs)

        _, self.anchor_points, _, self.stride_tensor = anchors_for_fpn_features(
            features,
            self.stride,
            self.grid_cell_size,
            self.grid_cell_offset,
            multiply_with_stride=False,
        )

        kpt_list: list[Tensor] = []
        bs = inputs[0].shape[0]  
        kpt_list = []
        for i in range(self.n_heads):
            kpt_pred = self.kpt_layers[i](inputs[i]).view(bs, self.nk, -1) # (bs, n_keypoints*3, h*w)
            kpt_list.append(kpt_pred)

        return features, cls_score_list, reg_distri_list, kpt_list

    def wrap(self, output: tuple[list[Tensor], list[Tensor], list[Tensor], list[Tensor]]) -> Packet[Tensor]:
        features, cls_score_list, reg_distri_list, kpt_list = output

        cls_tensor = torch.cat(
            [cls_score_list[i].flatten(2) for i in range(len(cls_score_list))], dim=2
        ).permute(0, 2, 1)
        reg_tensor = torch.cat(
            [reg_distri_list[i].flatten(2) for i in range(len(reg_distri_list))], dim=2
        ).permute(0, 2, 1)
        kpt_tensor = torch.cat(
            [kpt_list[i].flatten(2) for i in range(len(kpt_list))], dim=2
        ).permute(0, 2, 1) # (bs, h1*w1 + h2*w2 + h3*w3, n_keypoints*3)

        if self.export:
            kpt = self.kpts_decode(kpt_tensor)
            return {"out_cls": cls_tensor, "out_reg": reg_tensor, "kpt": kpt}

        if self.training:
            return {
                "features": features,
                "class_scores": [cls_tensor],
                "distributions": [reg_tensor],
                "keypoints_raw": [kpt_tensor],
            }

        pred_kpt = self.kpts_decode(kpt_tensor)
        detections = self._process_to_bbox_and_kps((features, cls_tensor, reg_tensor, pred_kpt))
        return {
            "boxes": [detection[:, :6] for detection in detections],
            "features": features,
            "class_scores": [cls_tensor],
            "distributions": [reg_tensor],
            "keypoints": [
                detection[:, 6:].reshape(-1, self.n_keypoints, 3) for detection in detections
            ],
            "keypoints_raw": [kpt_tensor],
        }

    def kpts_decode(self, kpts):
        """Decodes keypoints."""
        ndim = self.kpt_shape[1]
        y = kpts.clone() # (bs,h1*w1 + h2*w2 + h3*w3, n_keypoints*3) -> (bs, -1, n_keypoints*3)

        y[:, :, 2::3] = y[:, :,  2::3].sigmoid()

        anchor_points_transposed = self.anchor_points.transpose(0, 1)
        stride_tensor = self.stride_tensor.squeeze(-1) 

        stride_tensor = stride_tensor.view(1, -1, 1)
        anchor_points_x = anchor_points_transposed[0].view(1, -1, 1)
        anchor_points_y = anchor_points_transposed[1].view(1, -1, 1)

        y[:, :, 0::ndim] = (y[:, :, 0::ndim] * 2.0 + (anchor_points_x - 0.5)) * stride_tensor 
        y[:, :, 1::ndim] = (y[:, :, 1::ndim] * 2.0 + (anchor_points_y - 0.5)) * stride_tensor
        
        return y


    def _process_to_bbox_and_kps(
        self, output: tuple[list[Tensor], Tensor, Tensor]
    ) -> list[Tensor]:
        """Performs post-processing of the output and returns bboxs after NMS."""
        features, cls_score_list, reg_dist_list, keypoints = output
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
                keypoints,
            ],
            dim=-1,
        )

        return non_max_suppression(
            output_merged,
            n_classes=self.n_classes,
            conf_thres=self.conf_thres,
            iou_thres=self.iou_thres,
            bbox_format="xyxy",
            max_det=self.max_det,
            predicts_objectness=False,
        )
