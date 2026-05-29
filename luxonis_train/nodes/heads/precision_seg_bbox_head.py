from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from typing_extensions import override

from luxonis_train.nodes.blocks import ConvBlock, SegProto
from luxonis_train.tasks import Task, Tasks
from luxonis_train.typing import Packet
from luxonis_train.utils import (
    apply_bounding_box_to_masks,
    non_max_suppression,
)

from .precision_bbox_head import PrecisionBBoxHead


class PrecisionSegmentBBoxHead(PrecisionBBoxHead):
    task: Task = Tasks.INSTANCE_SEGMENTATION
    parser: str = "YOLOExtendedParser"

    def __init__(
        self,
        n_heads: Literal[2, 3, 4] = 3,
        n_masks: int = 32,
        n_proto: int = 64,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        max_det: int = 300,
        **kwargs,
    ):
        """
        Head for instance segmentation and object detection.
        Adapted from `Real-Time Flying Object Detection with YOLOv8 <https://arxiv.org/pdf/2305.09972>`_ and from `YOLOv6: A Single-Stage Object Detection Framework
        for Industrial Applications <https://arxiv.org/pdf/2209.02976.pdf>`_.

        Args:
            n_heads (Literal[2, 3, 4]): Number of output heads. Defaults to 3.
            n_masks (int): Number of masks.
            n_proto (int): Number of prototypes for segmentation.
            conf_thres (flaot): Confidence threshold for NMS.
            iou_thres (float): IoU threshold for NMS.
            max_det (int): Maximum number of detections retained after NMS.
        """
        super().__init__(
            n_heads=n_heads,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            max_det=max_det,
            **kwargs,
        )

        mid_channels = max(self.in_channels[0] // 4, n_masks)

        self.segmentation_heads = nn.ModuleList(
            nn.Sequential(
                ConvBlock(
                    in_channels=in_channels,
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
                nn.Conv2d(mid_channels, n_masks, 1, 1),
            )
            for in_channels in self.in_channels
        )

        self.proto = SegProto(self.in_channels[0], n_proto, n_masks)
        self.n_masks = n_masks

    def forward(self, inputs: list[Tensor]) -> Packet[Tensor]:
        prototypes = self.proto(inputs[0])
        mask_coefficients = [
            head(x)
            for head, x in zip(self.segmentation_heads, inputs, strict=True)
        ]

        features_list, classes_list, regressions_list = self.forward_heads(
            inputs
        )

        if self.export:
            pred_bboxes = self._construct_raw_bboxes(
                classes_list, regressions_list
            )
            return {
                "boundingbox": pred_bboxes,
                "masks": mask_coefficients,
                "prototypes": prototypes,
            }

        mask_coefficients = torch.cat(
            [
                coefficient.view(coefficient.size(0), self.n_masks, -1)
                for coefficient in mask_coefficients
            ],
            dim=2,
        )

        if self.training:
            return {
                "features": features_list,
                "prototypes": prototypes,
                "mask_coeficients": mask_coefficients,
            }

        pred_bboxes = self._prepare_bbox_inference_output(
            classes_list, regressions_list
        )
        preds_combined = torch.cat(
            [pred_bboxes, mask_coefficients.permute(0, 2, 1)], dim=-1
        )
        preds = non_max_suppression(
            preds_combined,
            n_classes=self.n_classes,
            conf_thres=self.conf_thres,
            iou_thres=self.iou_thres,
            bbox_format="xyxy",
            max_det=self.max_det,
            predicts_objectness=False,
        )

        results = {
            "features": features_list,
            "prototypes": prototypes,
            "mask_coeficients": mask_coefficients,
            "boundingbox": [],
            self.task.main_output: [],
        }

        for i, pred in enumerate(preds):
            height, width = self.original_in_shape[-2:]
            results[self.task.main_output].append(
                refine_and_apply_masks(
                    prototypes[i],
                    pred[:, 6:],
                    pred[:, :4],
                    height=height,
                    width=width,
                    upsample=True,
                )
            )
            results["boundingbox"].append(pred[:, :6])

        return results

    @property
    @override
    def export_output_names(self) -> list[str] | None:
        return self.get_output_names(
            [f"output{i + 1}_yolov8" for i in range(self.n_heads)]
            + [f"output{i + 1}_masks" for i in range(self.n_heads)]
            + ["protos_output"]
        )  # export names are applied on sorted output names


def refine_and_apply_masks(
    mask_prototypes: Tensor,
    predicted_masks: Tensor,
    bounding_boxes: Tensor,
    height: int,
    width: int,
    upsample: bool = False,
) -> Tensor:
    """Refine and apply masks to bounding boxes from the mask head.

    Args:
        mask_prototypes (Tensor): Prototype masks with shape
            ``[mask_dim, mask_height, mask_width]``.
        predicted_masks (Tensor): Predicted mask coefficients with shape
            ``[n_masks, mask_dim]``.
        bounding_boxes (Tensor): Bounding boxes with shape ``[n_masks, 4]``.
        height (int): Target image height.
        width (int): Target image width.
        upsample (bool): Whether to upsample masks to ``height`` and ``width``.
            Defaults to ``False``.

    Returns:
        Tensor: Binary mask tensor with shape ``[n_masks, height, width]``.
    """
    if predicted_masks.size(0) == 0 or bounding_boxes.size(0) == 0:
        return torch.zeros(
            0, height, width, dtype=torch.uint8, device=predicted_masks.device
        )

    channels, proto_h, proto_w = mask_prototypes.shape
    masks_combined = (
        predicted_masks @ mask_prototypes.float().view(channels, -1)
    ).view(-1, proto_h, proto_w)
    w_scale, h_scale = proto_w / width, proto_h / height
    scaled_boxes = bounding_boxes.clone()
    scaled_boxes[:, [0, 2]] *= w_scale
    scaled_boxes[:, [1, 3]] *= h_scale
    cropped_masks = apply_bounding_box_to_masks(masks_combined, scaled_boxes)
    if upsample:
        cropped_masks = F.interpolate(
            cropped_masks.unsqueeze(0),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    return (cropped_masks > 0).to(cropped_masks.dtype)
