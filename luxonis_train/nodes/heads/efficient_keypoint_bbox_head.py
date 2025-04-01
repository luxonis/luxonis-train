from typing import Literal

import torch
from loguru import logger
from torch import Tensor, nn

from luxonis_train.nodes.blocks import ConvModule
from luxonis_train.tasks import Tasks
from luxonis_train.typing import Packet
from luxonis_train.utils import (
    anchors_for_fpn_features,
    dist2bbox,
    non_max_suppression,
)

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

        self.nk = self.n_keypoints * 3

        mid_ch = max(self.in_channels[0] // 4, self.nk)
        self.kpt_layers = nn.ModuleList(
            nn.Sequential(
                ConvModule(x, mid_ch, 3, 1, 1, activation=nn.SiLU()),
                ConvModule(mid_ch, mid_ch, 3, 1, 1, activation=nn.SiLU()),
                nn.Conv2d(mid_ch, self.nk, 1, 1),
            )
            for x in self.in_channels
        )

        self._export_output_names = None

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
                f"output{i + 1}_yolov6" for i in range(self.n_heads)
            ] + [
                f"kpt_output{i + 1}" for i in range(self.n_heads)
            ]  # export names are applied on sorter output names

    def forward(
        self, inputs: list[Tensor]
    ) -> tuple[list[Tensor], list[Tensor], list[Tensor], list[Tensor]]:
        features, cls_score_list, reg_distri_list = super().forward(inputs)

        (_, self.anchor_points, self.n_anchors_list, self.stride_tensor) = (
            anchors_for_fpn_features(
                features,
                self.stride,
                self.grid_cell_size,
                self.grid_cell_offset,
                multiply_with_stride=False,
            )
        )

        kpt_list: list[Tensor] = []
        for i in range(self.n_heads):
            kpt_pred = self.kpt_layers[i](inputs[i])
            kpt_list.append(kpt_pred)

        return features, cls_score_list, reg_distri_list, kpt_list

    def wrap(
        self,
        output: tuple[list[Tensor], list[Tensor], list[Tensor], list[Tensor]],
    ) -> Packet[Tensor]:
        features, cls_score_list, reg_distri_list, kpt_list = output
        bs = features[0].shape[0]
        if self.export:
            det_outputs: list[Tensor] = []
            kpt_outputs: list[Tensor] = []
            for i, (out_cls, out_reg, out_kpt) in enumerate(
                zip(cls_score_list, reg_distri_list, kpt_list, strict=True)
            ):
                conf, _ = out_cls.max(1, keepdim=True)
                out = torch.cat([out_reg, conf, out_cls], dim=1)
                det_outputs.append(out)
                kpt_outputs.append(
                    self._dist2kpts(
                        out_kpt.view(bs, self.nk, -1),
                        bs,
                        i,
                        apply_sigmoid=False,
                    )
                )
            return {"boundingbox": det_outputs, "keypoints": kpt_outputs}

        cls_tensor = torch.cat(
            [cls_score_list[i].flatten(2) for i in range(len(cls_score_list))],
            dim=2,
        ).permute(0, 2, 1)
        reg_tensor = torch.cat(
            [
                reg_distri_list[i].flatten(2)
                for i in range(len(reg_distri_list))
            ],
            dim=2,
        ).permute(0, 2, 1)
        kpt_tensor = torch.cat(
            [
                kpt_list[i].view(bs, self.nk, -1).flatten(2)
                for i in range(len(kpt_list))
            ],
            dim=2,
        ).permute(0, 2, 1)

        if self.training:
            return {
                "features": features,
                "class_scores": cls_tensor,
                "distributions": reg_tensor,
                "keypoints_raw": kpt_tensor,
            }
        pred_kpt = torch.cat(
            [
                self._dist2kpts(kpt_list[i].view(bs, self.nk, -1), bs, i)
                for i in range(len(kpt_list))
            ],
            dim=2,
        ).permute(0, 2, 1)
        detections = self._process_to_bbox_and_kps(
            (features, cls_tensor, reg_tensor, pred_kpt)
        )
        return {
            "boundingbox": [detection[:, :6] for detection in detections],
            "keypoints": [
                detection[:, 6:].reshape(-1, self.n_keypoints, 3)
                for detection in detections
            ],
            "features": features,
            "class_scores": cls_tensor,
            "distributions": reg_tensor,
            "keypoints_raw": kpt_tensor,
        }

    def _dist2kpts(
        self,
        kpts: Tensor,
        batch_size: int,
        index: int,
        apply_sigmoid: bool = True,
    ) -> Tensor:
        """Decodes keypoints."""
        anchors = self.anchor_points.split(self.n_anchors_list, dim=0)
        kpt_predictions = kpts.view(batch_size, self.n_keypoints, 3, -1)
        grid_coords = (
            kpt_predictions[:, :, :2] * 2.0
            + (anchors[index].transpose(1, 0) - 0.5)
        ) * self.stride[index]

        conf_scores = kpt_predictions[:, :, 2:3]
        if apply_sigmoid:
            conf_scores = conf_scores.sigmoid()

        decoded_kpts = torch.cat(
            (
                grid_coords,
                conf_scores,
            ),
            2,
        )
        return decoded_kpts.view(batch_size, self.nk, -1)

    def _process_to_bbox_and_kps(
        self, output: tuple[list[Tensor], Tensor, Tensor, Tensor]
    ) -> list[Tensor]:
        """Performs post-processing of the output and returns bboxs
        after NMS."""
        features, cls_score_list, reg_dist_list, keypoints = output

        pred_bboxes = dist2bbox(
            reg_dist_list, self.anchor_points, out_format="xyxy"
        )

        pred_bboxes *= self.stride_tensor
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

    def get_custom_head_config(self) -> dict:
        """Returns custom head configuration.

        @rtype: dict
        @return: Custom head configuration.
        """
        return {
            "subtype": "yolov6r2",
            "iou_threshold": self.iou_thres,
            "conf_threshold": self.conf_thres,
            "max_det": self.max_det,
            "n_keypoints": self.n_keypoints,
        }
