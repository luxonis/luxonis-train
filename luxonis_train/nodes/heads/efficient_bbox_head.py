from typing import Literal

import torch
from loguru import logger
from torch import Tensor, nn

from luxonis_train.nodes.blocks import EfficientDecoupledBlock
from luxonis_train.nodes.heads import BaseHead
from luxonis_train.tasks import Tasks
from luxonis_train.typing import Packet
from luxonis_train.utils import (
    anchors_for_fpn_features,
    dist2bbox,
    non_max_suppression,
)


class EfficientBBoxHead(
    BaseHead[list[Tensor], tuple[list[Tensor], list[Tensor], list[Tensor]]],
):
    task = Tasks.BOUNDINGBOX
    parser = "YOLO"

    in_channels: list[int]

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
        super().__init__(**kwargs)

        self.n_heads = n_heads

        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det

        self.stride = self._fit_stride_to_n_heads()
        self.grid_cell_offset = 0.5
        self.grid_cell_size = 5.0

        self.heads = nn.ModuleList()
        if len(self.in_channels) < self.n_heads:
            logger.warning(
                f"Head '{self.name}' was set to use {self.n_heads} heads, "
                f"but received only {len(self.in_channels)} inputs. "
                f"Changing number of heads to {len(self.in_channels)}."
            )
            self.n_heads = len(self.in_channels)
        for i in range(self.n_heads):
            curr_head = EfficientDecoupledBlock(
                n_classes=self.n_classes,
                in_channels=self.in_channels[i],
            )
            self.heads.append(curr_head)

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

        self.check_export_output_names()

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

    def get_variant_weights(self, initialize_weights: bool) -> str | None:
        if self.in_channels == [32, 64, 128]:  # light predefined model
            if initialize_weights:
                return "https://github.com/luxonis/luxonis-train/releases/download/v0.2.1-beta/efficientbbox_head_n_coco.ckpt"
            else:
                return "https://github.com/luxonis/luxonis-train/releases/download/v0.1.0-beta/efficientbbox_head_n_coco.ckpt"
        elif self.in_channels == [64, 128, 256]:  # medium predefined model
            if initialize_weights:
                return "https://github.com/luxonis/luxonis-train/releases/download/v0.2.1-beta/efficientbbox_head_s_coco.ckpt"
            else:
                return None
        elif self.in_channels == [128, 256, 512]:  # heavy predefined model
            if initialize_weights:
                return "https://github.com/luxonis/luxonis-train/releases/download/v0.2.1-beta/efficientbbox_head_l_coco.ckpt"
            else:
                return None
        else:
            return None

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
                f"output{i + 1}_yolov6r2" for i in range(self.n_heads)
            ]

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
            outputs: list[Tensor] = []
            for out_cls, out_reg in zip(
                cls_score_list, reg_distri_list, strict=True
            ):
                conf, _ = out_cls.max(1, keepdim=True)
                out = torch.cat([out_reg, conf, out_cls], dim=1)
                outputs.append(out)
            return {"boundingbox": outputs}

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

        if self.training:
            return {
                "features": features,
                "class_scores": cls_tensor,
                "distributions": reg_tensor,
            }

        else:
            boxes = self._process_to_bbox((features, cls_tensor, reg_tensor))
            return {
                "boundingbox": boxes,
                "features": features,
                "class_scores": cls_tensor,
                "distributions": reg_tensor,
            }

    def _fit_stride_to_n_heads(self) -> Tensor:
        """Returns correct stride for number of heads and attach
        index."""
        stride = torch.tensor(
            [
                round(
                    self.original_in_shape[1] / x[2]
                )  # Ensure proper rounding of the stride
                for x in self.in_sizes[: self.n_heads]
            ],
            dtype=torch.int,
        )
        return stride

    def _process_to_bbox(
        self, output: tuple[list[Tensor], Tensor, Tensor]
    ) -> list[Tensor]:
        """Performs post-processing of the output and returns bboxs
        after NMS."""
        features, cls_score_list, reg_dist_list = output
        _, anchor_points, _, stride_tensor = anchors_for_fpn_features(
            features,
            self.stride,
            self.grid_cell_size,
            self.grid_cell_offset,
            multiply_with_stride=False,
        )

        pred_bboxes = dist2bbox(
            reg_dist_list, anchor_points, out_format="xyxy"
        )

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
        }
