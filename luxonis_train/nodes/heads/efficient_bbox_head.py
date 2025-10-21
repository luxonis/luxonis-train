import math
from collections.abc import Iterable
from typing import Literal, cast

import torch
from luxonis_ml.typing import Params
from torch import Tensor, nn
from typing_extensions import override

from luxonis_train.nodes.blocks import EfficientDecoupledBlock
from luxonis_train.nodes.heads.base_detection_head import BaseDetectionHead
from luxonis_train.tasks import Tasks
from luxonis_train.typing import Packet
from luxonis_train.utils import (
    anchors_for_fpn_features,
    dist2bbox,
    non_max_suppression,
)


class EfficientBBoxHead(BaseDetectionHead):
    task = Tasks.BOUNDINGBOX

    def __init__(
        self,
        n_heads: Literal[2, 3, 4] = 3,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        max_det: int = 300,
        bias_init_p: float = 1e-2,
        **kwargs,
    ):
        """Head for object detection.

        Adapted from U{YOLOv6: A Single-Stage Object Detection Framework
        for Industrial Applications
        <https://arxiv.org/pdf/2209.02976.pdf>}.
        @type n_heads: Literal[2,3,4]
        @param n_heads: Number of output heads. Defaults to 3. B{Note:}
            Should be same also on neck in most cases.
        @type conf_thres: float
        @param conf_thres: Threshold for confidence. Defaults to
            C{0.25}.
        @type iou_thres: float
        @param iou_thres: Threshold for IoU. Defaults to C{0.45}.
        @type max_det: int
        @param max_det: Maximum number of detections retained after NMS.
            Defaults to C{300}.
        """
        super().__init__(
            n_heads=n_heads,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            max_det=max_det,
            **kwargs,
        )

        self.grid_cell_offset = 0.5
        self.grid_cell_size = 5.0

        self.heads = cast(list[EfficientDecoupledBlock], nn.ModuleList())

        for i in range(self.n_heads):
            self.heads.append(
                EfficientDecoupledBlock(
                    in_channels=self.in_channels[i], n_classes=self.n_classes
                )
            )
        self.bias_init_p = bias_init_p

    @override
    def initialize_weights(self, method: str | None = None) -> None:
        super().initialize_weights(method)
        for head in self.heads:
            data = [
                (
                    head.class_branch[-1],
                    -math.log((1 - self.bias_init_p) / self.bias_init_p),
                ),
                (head.regression_branch[-1], 1.0),
            ]
            for module, fill_value in data:
                assert isinstance(module, nn.Conv2d)
                assert module.bias is not None

                nn.init.constant_(module.weight, 0)
                nn.init.constant_(module.bias, fill_value)

    @override
    def load_checkpoint(
        self, path: str | None = None, strict: bool = False
    ) -> None:
        return super().load_checkpoint(path, strict=strict)

    def forward(self, inputs: list[Tensor]) -> Packet[Tensor]:
        features_list, classes_list, regressions_list = self._forward(inputs)

        if self.export:
            return self._wrap_export(classes_list, regressions_list)

        class_scores = self._postprocess(classes_list)
        distributions = self._postprocess(regressions_list)

        if self.training:
            return {
                "features": features_list,
                "class_scores": class_scores,
                "distributions": distributions,
            }

        _, anchor_points, _, stride_tensor = anchors_for_fpn_features(
            features_list,
            self.stride,
            self.grid_cell_size,
            self.grid_cell_offset,
            multiply_with_stride=False,
        )
        bboxes = self._postprocess_detections(
            features_list,
            class_scores,
            distributions,
            anchor_points,
            stride_tensor,
        )
        return {
            self.task.main_output: bboxes,
            "features": features_list,
            "class_scores": class_scores,
            "distributions": distributions,
        }

    @staticmethod
    def _wrap_export(
        classes_list: list[Tensor], regressions_list: list[Tensor]
    ) -> Packet[Tensor]:
        bboxes: list[Tensor] = []
        for classes, regressions in zip(
            classes_list, regressions_list, strict=True
        ):
            conf, _ = classes.max(1, keepdim=True)
            out = torch.cat([regressions, conf, classes], dim=1)
            bboxes.append(out)
        return {"boundingbox": bboxes}

    @staticmethod
    def _postprocess(outputs: Iterable[Tensor]) -> Tensor:
        return torch.cat([out.flatten(2) for out in outputs], dim=2).permute(
            0, 2, 1
        )

    @override
    def get_weights_url(self) -> str:
        return "{github}/efficientbbox_head_{variant}_coco.ckpt"

    @property
    @override
    def export_output_names(self) -> list[str] | None:
        return self.get_output_names(
            [f"output{i + 1}_yolov6r2" for i in range(self.n_heads)]
        )

    def _postprocess_detections(
        self,
        features: list[Tensor],
        class_scores: Tensor,
        distributions: Tensor,
        anchor_points: Tensor,
        stride_tensor: Tensor,
        *,
        tail: list[Tensor] | None = None,
    ) -> list[Tensor]:
        """Performs post-processing of the output and returns bboxs
        after NMS."""

        tail = tail or []
        bboxes = dist2bbox(distributions, anchor_points, out_format="xyxy")

        bboxes *= stride_tensor
        output_merged = torch.cat(
            [
                bboxes,
                torch.ones(
                    (features[-1].shape[0], bboxes.shape[1], 1),
                    dtype=bboxes.dtype,
                    device=bboxes.device,
                ),
                class_scores,
                *tail,
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

    @override
    def get_custom_head_config(self) -> Params:
        """Returns custom head configuration.

        @rtype: dict
        @return: Custom head configuration.
        """
        return super().get_custom_head_config() | {"subtype": "yolov6r2"}
