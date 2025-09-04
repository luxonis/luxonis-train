from typing import Literal

import torch
from luxonis_ml.typing import Params
from torch import Tensor, nn
from typing_extensions import override

from luxonis_train.nodes.blocks import ConvBlock
from luxonis_train.tasks import Tasks
from luxonis_train.typing import Packet
from luxonis_train.utils import anchors_for_fpn_features

from .efficient_bbox_head import EfficientBBoxHead


class EfficientKeypointBBoxHead(EfficientBBoxHead):
    parser = "YOLOExtendedParser"
    task = Tasks.INSTANCE_KEYPOINTS

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

        mid_channels = max(self.in_channels[0] // 4, self.n_keypoints_flat)
        self.keypoint_heads = nn.ModuleList(
            nn.Sequential(
                ConvBlock(
                    in_channels=self.in_channels[i],
                    out_channels=mid_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    activation=nn.SiLU(),
                ),
                ConvBlock(
                    in_channels=mid_channels,
                    out_channels=mid_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    activation=nn.SiLU(),
                ),
                nn.Conv2d(
                    in_channels=mid_channels,
                    out_channels=self.n_keypoints_flat,
                    kernel_size=1,
                    stride=1,
                ),
            )
            for i in range(len(self.heads))
        )

    def forward(self, inputs: list[Tensor]) -> Packet[Tensor]:
        features_list, classes_list, regressions_list = super()._forward(
            inputs
        )
        keypoints_list: list[Tensor] = []

        for head, x in zip(self.keypoint_heads, inputs, strict=True):
            keypoints_list.append(head(x))

        bs = features_list[0].shape[0]
        if self.export:
            packet = self._wrap_export(classes_list, regressions_list)
            keypoints = []
            for i, keypoint in enumerate(keypoints_list):
                keypoints.append(
                    self._distributions_to_keypoints(
                        keypoint.view(bs, self.n_keypoints_flat, -1),
                        features_list,
                        bs,
                        i,
                        apply_sigmoid=False,
                    )
                )
            return packet | {"keypoints": keypoints}

        class_scores = self._postprocess(classes_list)
        distributions = self._postprocess(regressions_list)
        keypoints_raw = self._postprocess(
            out.view(bs, self.n_keypoints_flat, -1) for out in keypoints_list
        )

        if self.training:
            return {
                "features": features_list,
                "class_scores": class_scores,
                "distributions": distributions,
                "keypoints_raw": keypoints_raw,
            }

        pred_keypoints = torch.cat(
            [
                self._distributions_to_keypoints(
                    keypoint.view(bs, self.n_keypoints_flat, -1),
                    features_list,
                    bs,
                    i,
                )
                for i, keypoint in enumerate(keypoints_list)
            ],
            dim=2,
        ).permute(0, 2, 1)

        _, anchor_points, _, stride_tensor = anchors_for_fpn_features(
            features_list,
            self.stride,
            self.grid_cell_size,
            self.grid_cell_offset,
            multiply_with_stride=False,
        )
        boxes, kpts = self._postprocess_keypoint_detections(
            features_list,
            class_scores,
            distributions,
            pred_keypoints,
            anchor_points,
            stride_tensor,
        )
        return {
            "boundingbox": boxes,
            "keypoints": kpts,
            "features": features_list,
            "class_scores": class_scores,
            "distributions": distributions,
            "keypoints_raw": keypoints_raw,
        }

    @property
    @override
    def export_output_names(self) -> list[str] | None:
        return self.get_output_names(
            [f"output{i + 1}_yolov6" for i in range(self.n_heads)]
            + [f"kpt_output{i + 1}" for i in range(self.n_heads)]
        )

    @override
    def get_custom_head_config(self) -> Params:
        """Returns custom head configuration.

        @rtype: dict
        @return: Custom head configuration.
        """
        return super().get_custom_head_config() | {
            "n_keypoints": self.n_keypoints
        }

    def _distributions_to_keypoints(
        self,
        keypoints: Tensor,
        features: list[Tensor],
        batch_size: int,
        index: int,
        apply_sigmoid: bool = True,
    ) -> Tensor:
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

    def _postprocess_keypoint_detections(
        self,
        features: list[Tensor],
        class_scores: Tensor,
        distributions: Tensor,
        pred_keypoints: Tensor,
        anchor_points: Tensor,
        stride_tensor: Tensor,
    ) -> tuple[list[Tensor], list[Tensor]]:
        """Performs post-processing of the output and returns bboxs
        after NMS."""

        detections = super()._postprocess_detections(
            features,
            class_scores,
            distributions,
            anchor_points,
            stride_tensor,
            tail=[pred_keypoints],
        )

        bboxes = [detection[:, :6] for detection in detections]
        keypoints = [
            detection[:, 6:].reshape(-1, self.n_keypoints, 3)
            for detection in detections
        ]
        return bboxes, keypoints
