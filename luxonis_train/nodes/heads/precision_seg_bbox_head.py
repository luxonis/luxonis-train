from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from luxonis_train.enums import TaskType
from luxonis_train.nodes.blocks import ConvModule, SegProto
from luxonis_train.utils import (
    Packet,
    apply_bounding_box_to_masks,
    non_max_suppression,
)

from .precision_bbox_head import PrecisionBBoxHead


class PrecisionSegmentBBoxHead(PrecisionBBoxHead):
    tasks: list[TaskType] = [
        TaskType.INSTANCE_SEGMENTATION,
        TaskType.BOUNDINGBOX,
    ]

    def __init__(
        self,
        n_heads: Literal[2, 3, 4] = 3,
        n_masks: int = 32,
        n_proto: int = 256,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        max_det: int = 300,
        **kwargs: Any,
    ):
        """
        Head for instance segmentation and object detection.
        Adapted from U{Real-Time Flying Object Detection with YOLOv8
        <https://arxiv.org/pdf/2305.09972>}

        @type n_heads: Literal[2, 3, 4]
        @param n_heads: Number of output heads. Defaults to 3.
        @type n_masks: int
        @param n_masks: Number of masks.
        @type n_proto: int
        @param n_proto: Number of prototypes for segmentation.
        @type conf_thres: flaot
        @param conf_thres: Confidence threshold for NMS.
        @type iou_thres: float
        @param iou_thres: IoU threshold for NMS.
        @type max_det: int
        @param max_det: Maximum number of detections retained after NMS.
        """
        super().__init__(
            n_heads=n_heads,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            max_det=max_det,
            **kwargs,
        )

        self.n_masks = n_masks
        self.n_proto = n_proto

        self.proto = SegProto(self.in_channels[0], self.n_proto, self.n_masks)

        mid_ch = max(self.in_channels[0] // 4, self.n_masks)
        self.mask_layers = nn.ModuleList(
            nn.Sequential(
                ConvModule(x, mid_ch, 3, 1, 1, activation=nn.SiLU()),
                ConvModule(mid_ch, mid_ch, 3, 1, 1, activation=nn.SiLU()),
                nn.Conv2d(mid_ch, self.n_masks, 1, 1),
            )
            for x in self.in_channels
        )

        self._export_output_names = None

    def forward(
        self, inputs: list[Tensor]
    ) -> tuple[list[Tensor], list[Tensor], list[Tensor], list[Tensor]]:
        prototypes = self.proto(inputs[0])
        bs = prototypes.shape[0]
        mask_coefficients = torch.cat(
            [
                self.mask_layers[i](inputs[i]).view(bs, self.n_masks, -1)
                for i in range(self.n_heads)
            ],
            dim=2,
        )
        det_outs = super().forward(inputs)

        return det_outs, prototypes, mask_coefficients

    def wrap(
        self, output: tuple[list[Tensor], Tensor, Tensor]
    ) -> Packet[Tensor]:
        det_feats, prototypes, mask_coefficients = output
        if self.training:
            return {
                "features": det_feats,
                "prototypes": prototypes,
                "mask_coeficients": mask_coefficients,
            }
        if self.export:
            {
                self.task: (
                    torch.cat([det_feats, mask_coefficients], 1),
                    prototypes,
                )
            }
        pred_bboxes = self._inference(det_feats, mask_coefficients)
        preds = non_max_suppression(
            pred_bboxes,
            n_classes=self.n_classes,
            conf_thres=self.conf_thres,
            iou_thres=self.iou_thres,
            bbox_format="xyxy",
            max_det=self.max_det,
            predicts_objectness=False,
        )

        results = {
            "features": det_feats,
            "prototypes": prototypes,
            "mask_coeficients": mask_coefficients,
            "boundingbox": [],
            "instance_segmentation": [],
        }

        for i, pred in enumerate(
            preds
        ):  # TODO: Investigate low seg loss but wrong masks
            results["instance_segmentation"].append(
                refine_and_apply_masks(
                    prototypes[i],
                    pred[:, 6:],
                    pred[:, :4],
                    self.original_in_shape[-2:],
                    upsample=True,
                )
            )
            results["boundingbox"].append(pred[:, :6])

        return results


def refine_and_apply_masks(
    mask_prototypes,
    predicted_masks,
    bounding_boxes,
    target_shape,
    upsample=False,
):
    """Refine and apply masks to bounding boxes based on the mask head
    outputs.

    @type mask_prototypes: torch.Tensor
    @param mask_prototypes: Tensor of shape [mask_dim, mask_height,
        mask_width].
    @type predicted_masks: torch.Tensor
    @param predicted_masks: Tensor of shape [num_masks, mask_dim], where
        num_masks is the number of detected masks.
    @type bounding_boxes: torch.Tensor
    @param bounding_boxes: Tensor of shape [num_masks, 4], containing
        bounding box coordinates.
    @type target_shape: tuple
    @param target_shape: Tuple (height, width) representing the
        dimensions of the original image.
    @type upsample: bool
    @param upsample: If True, upsample the masks to the target image
        dimensions. Default is False.
    @rtype: torch.Tensor
    @return: A binary mask tensor of shape [num_masks, height, width],
        where the masks are cropped according to their respective
        bounding boxes.
    """
    if predicted_masks.size(0) == 0 or bounding_boxes.size(0) == 0:
        img_h, img_w = target_shape
        return torch.zeros(0, img_h, img_w, dtype=torch.uint8)

    channels, proto_h, proto_w = mask_prototypes.shape
    img_h, img_w = target_shape
    masks_combined = (
        predicted_masks @ mask_prototypes.float().view(channels, -1)
    ).view(-1, proto_h, proto_w)
    w_scale, h_scale = proto_w / img_w, proto_h / img_h
    scaled_boxes = bounding_boxes.clone()
    scaled_boxes[:, [0, 2]] *= w_scale
    scaled_boxes[:, [1, 3]] *= h_scale
    cropped_masks = apply_bounding_box_to_masks(masks_combined, scaled_boxes)
    if upsample:
        cropped_masks = F.interpolate(
            cropped_masks.unsqueeze(0),
            size=target_shape,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    return (cropped_masks > 0).to(cropped_masks.dtype)
