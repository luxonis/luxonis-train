from typing import Literal

import torch
from torch import Tensor, nn
from typing_extensions import override

from luxonis_train.nodes.blocks import ConvModule
from luxonis_train.tasks import Tasks
from luxonis_train.typing import Packet
from luxonis_train.utils import anchors_for_fpn_features

from .efficient_bbox_head import EfficientBBoxHead


class EfficientKeypointBBoxHead(EfficientBBoxHead):
    task = Tasks.INSTANCE_KEYPOINTS
    parser: str = "YOLOExtendedParser"

    def __init__(
        self,
        n_heads: Literal[2, 3, 4] = 3,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        max_det: int = 300,
        **kwargs,
    ):
        """Head for object and keypoint detection.

        Adapted from U{YOLOv6: A Single-Stage Object Detection Framework for Industrial
        Applications<https://arxiv.org/pdf/2209.02976.pdf>}.

        @param n_heads: Number of output heads. Defaults to C{3}.
            B{Note:} Should be same also on neck in most cases.
        @type n_heads: int

        @param conf_thres: Threshold for confidence. Defaults to C{0.25}.
        @type conf_thres: float

        @param iou_thres: Threshold for IoU. Defaults to C{0.45}.
        @type iou_thres: float

        @param max_det: Maximum number of detections retained after NMS. Defaults to C{300}.
        @type max_det: int
        """
        super().__init__(
            n_heads=n_heads,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            max_det=max_det,
            **kwargs,
        )

        self.n_keypoints_flat = self.n_keypoints * 3

        mid_ch = max(self.in_channels[0] // 4, self.n_keypoints_flat)
        self.kpt_layers = nn.ModuleList(
            nn.Sequential(
                ConvModule(x, mid_ch, 3, 1, 1, activation=nn.SiLU()),
                ConvModule(mid_ch, mid_ch, 3, 1, 1, activation=nn.SiLU()),
                nn.Conv2d(mid_ch, self.n_keypoints_flat, 1, 1),
            )
            for x in self.in_channels
        )

    def forward(
        self, inputs: list[Tensor]
    ) -> tuple[list[Tensor], list[Tensor], list[Tensor], list[Tensor]]:
        features, cls_score_list, reg_distri_list = super().forward(inputs)

        kpt_list: list[Tensor] = []
        for i in range(self.n_heads):
            kpt_pred = self.kpt_layers[i](inputs[i])
            kpt_list.append(kpt_pred)

        return features, cls_score_list, reg_distri_list, kpt_list

    @override
    def wrap(
        self,
        output: tuple[list[Tensor], list[Tensor], list[Tensor], list[Tensor]],
    ) -> Packet[Tensor]:
        features, cls_score_list, reg_distri_list, kpt_list = output
        bs = features[0].shape[0]
        if self.export:
            packet = self._wrap_export(cls_score_list, reg_distri_list)
            keypoints = []
            for i, kpt in enumerate(kpt_list):
                keypoints.append(
                    self._dist2kpts(
                        kpt.view(bs, self.n_keypoints_flat, -1),
                        features,
                        bs,
                        i,
                        apply_sigmoid=False,
                    )
                )
            return packet | {"keypoints": keypoints}

        cls_tensor = self._postprocess(cls_score_list)
        reg_tensor = self._postprocess(reg_distri_list)
        kpt_tensor = self._postprocess(
            out.view(bs, self.n_keypoints_flat, -1) for out in kpt_list
        )

        if self.training:
            return {
                "features": features,
                "class_scores": cls_tensor,
                "distributions": reg_tensor,
                "keypoints_raw": kpt_tensor,
            }

        pred_kpt = torch.cat(
            [
                self._dist2kpts(
                    kpt.view(bs, self.n_keypoints_flat, -1), features, bs, i
                )
                for i, kpt in enumerate(kpt_list)
            ],
            dim=2,
        ).permute(0, 2, 1)
        _, anchor_points, _, stride_tensor = anchors_for_fpn_features(
            features,
            self.stride,
            self.grid_cell_size,
            self.grid_cell_offset,
            multiply_with_stride=False,
        )
        boxes, kpts = self._postprocess_detections(
            features,
            cls_tensor,
            reg_tensor,
            pred_kpt,
            anchor_points,
            stride_tensor,
        )
        return {
            "boundingbox": boxes,
            "keypoints": kpts,
            "features": features,
            "class_scores": cls_tensor,
            "distributions": reg_tensor,
            "keypoints_raw": kpt_tensor,
        }

    @property
    @override
    def export_output_names(self) -> list[str] | None:
        export_names = super().export_output_names
        if self._check_output_names(export_names, self.n_heads):
            return export_names

        return [f"output{i + 1}_yolov6" for i in range(self.n_heads)] + [
            f"kpt_output{i + 1}" for i in range(self.n_heads)
        ]

    def _dist2kpts(
        self,
        keypoints: Tensor,
        features: list[Tensor],
        batch_size: int,
        index: int,
        apply_sigmoid: bool = True,
    ) -> Tensor:
        """Decodes keypoints."""

        _, anchor_points, n_anchors_list, _ = anchors_for_fpn_features(
            features,
            self.stride,
            self.grid_cell_size,
            self.grid_cell_offset,
            multiply_with_stride=False,
        )
        anchors = anchor_points.split(n_anchors_list, dim=0)
        keypoints = keypoints.view(batch_size, self.n_keypoints, 3, -1)
        grid_coords = (
            keypoints[:, :, :2] * 2.0 + (anchors[index].transpose(1, 0) - 0.5)
        ) * self.stride[index]

        conf_scores = keypoints[:, :, 2:3]

        if apply_sigmoid:
            conf_scores = conf_scores.sigmoid()

        return torch.cat((grid_coords, conf_scores), dim=2).view(
            batch_size, self.n_keypoints_flat, -1
        )

    @override
    def _postprocess_detections(
        self,
        features: list[Tensor],
        cls_tensor: Tensor,
        reg_tensor: Tensor,
        keypoints: Tensor,
        anchor_points: Tensor,
        stride_tensor: Tensor,
    ) -> tuple[list[Tensor], list[Tensor]]:
        """Performs post-processing of the output and returns bboxs
        after NMS."""

        detections = super()._postprocess_detections(
            features,
            cls_tensor,
            reg_tensor,
            anchor_points,
            stride_tensor,
            tail=[keypoints],
        )

        boxes = [detection[:, :6] for detection in detections]
        kpts = [
            detection[:, 6:].reshape(-1, self.n_keypoints, 3)
            for detection in detections
        ]
        return boxes, kpts
