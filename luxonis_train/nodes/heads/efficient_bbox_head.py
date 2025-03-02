from collections.abc import Iterable
from typing import Literal

import torch
from loguru import logger
from torch import Tensor, nn
from typing_extensions import override

from luxonis_train.nodes.blocks import EfficientDecoupledBlock
from luxonis_train.nodes.heads.precision_bbox_head import BaseDetectionHead
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
        download_weights: bool = False,
        initialize_weights: bool = True,
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
        @type download_weights: bool
        @param download_weights: If True download weights from COCO.
            Defaults to False.
        @type initialize_weights: bool
        @param initialize_weights: If True, initialize weights.
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

        self.heads = nn.ModuleList()
        # TODO: What to do if inputs are longer than heads? Create
        # more heads or discard some inputs? If discard, do we discared
        # the deeper features or the shallower ones?
        if len(self.in_channels) < self.n_heads:
            logger.warning(
                f"Head '{self.name}' was set to use {self.n_heads} heads, "
                f"but received {len(self.in_channels)} inputs. "
                f"Changing number of heads to {len(self.in_channels)}."
            )
            self.n_heads = len(self.in_channels)
        for i in range(self.n_heads):
            self.heads.append(
                EfficientDecoupledBlock(
                    in_channels=self.in_channels[i], n_classes=self.n_classes
                )
            )

        if initialize_weights:
            self.initialize_weights()

        if (
            download_weights and self.name == "EfficientBBoxHead"
        ):  # skip download on classes that inherit this one
            weights_path = self.get_variant_weights(initialize_weights)
            if weights_path:
                self.load_checkpoint(path=weights_path, strict=False)
            else:
                logger.warning(
                    f"No checkpoint available for {self.name}, skipping."
                )

    def forward(
        self, inputs: list[Tensor]
    ) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
        features_list: list[Tensor] = []
        classes_list: list[Tensor] = []
        regressions_list: list[Tensor] = []

        # FIXME: strict=True, related to the TODO above
        for head, x in zip(self.heads, inputs, strict=False):
            features, classes, regressions = head(x)
            features_list.append(features)
            classes_list.append(torch.sigmoid(classes))
            regressions_list.append(regressions)

        return features_list, classes_list, regressions_list

    @override
    def wrap(
        self, output: tuple[list[Tensor], list[Tensor], list[Tensor]]
    ) -> Packet[Tensor]:
        features, classes_list, regressions_list = output

        if self.export:
            return self._wrap_export(classes_list, regressions_list)

        class_scores = self._postprocess(classes_list)
        distributions = self._postprocess(regressions_list)

        if self.training:
            return {
                "features": features,
                "class_scores": class_scores,
                "distributions": distributions,
            }

        _, anchor_points, _, stride_tensor = anchors_for_fpn_features(
            features,
            self.stride,
            self.grid_cell_size,
            self.grid_cell_offset,
            multiply_with_stride=False,
        )
        boxes = self._postprocess_detections(
            features, class_scores, distributions, anchor_points, stride_tensor
        )
        return {
            "boundingbox": boxes,
            "features": features,
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

    def get_variant_weights(self, initialize_weights: bool) -> str | None:
        if self.in_channels == [32, 64, 128]:  # light predefined model
            if initialize_weights:
                return "https://github.com/luxonis/luxonis-train/releases/download/v0.2.1-beta/efficientbbox_head_n_coco.ckpt"
            return "https://github.com/luxonis/luxonis-train/releases/download/v0.1.0-beta/efficientbbox_head_n_coco.ckpt"
        if self.in_channels == [64, 128, 256]:  # medium predefined model
            if initialize_weights:
                return "https://github.com/luxonis/luxonis-train/releases/download/v0.2.1-beta/efficientbbox_head_s_coco.ckpt"
            return None
        if self.in_channels == [128, 256, 512]:  # heavy predefined model
            if initialize_weights:
                return "https://github.com/luxonis/luxonis-train/releases/download/v0.2.1-beta/efficientbbox_head_l_coco.ckpt"
            return None
        return None

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
    def get_custom_head_config(self) -> dict:
        """Returns custom head configuration.

        @rtype: dict
        @return: Custom head configuration.
        """
        return super().get_custom_head_config() | {"subtype": "yolov6"}
