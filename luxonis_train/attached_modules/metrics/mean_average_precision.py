from typing import Any

import torch
import torchmetrics.detection as detection
from torch import Tensor
from torchvision.ops import box_convert

from luxonis_train.enums import TaskType
from luxonis_train.utils import Labels, Packet

from .base_metric import BaseMetric


class MeanAveragePrecision(
    BaseMetric[list[dict[str, Tensor]], list[dict[str, Tensor]]]
):
    """Compute the Mean-Average-Precision (mAP) and Mean-Average-Recall
    (mAR) for object detection predictions and instance segmentation.

    Adapted from U{Mean-Average-Precision (mAP) and Mean-Average-Recall
    (mAR)
    <https://lightning.ai/docs/torchmetrics/stable/detection/mean_average_precision.html>}.
    """

    supported_tasks: list[TaskType] = [
        TaskType.BOUNDINGBOX,
        TaskType.INSTANCE_SEGMENTATION,
    ]

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.is_segmentation = (
            TaskType.INSTANCE_SEGMENTATION in self.node.tasks
        )

        if self.is_segmentation:
            iou_type = ("bbox", "segm")
        else:
            iou_type = "bbox"

        self.metric = detection.MeanAveragePrecision(iou_type=iou_type)

    def update(
        self,
        outputs: list[dict[str, Tensor]],
        labels: list[dict[str, Tensor]],
    ):
        self.metric.update(outputs, labels)

    def prepare(
        self, inputs: Packet[Tensor], labels: Labels
    ) -> tuple[list[dict[str, Tensor]], list[dict[str, Tensor]]]:
        box_label = self.get_label(labels, TaskType.BOUNDINGBOX)
        mask_label = (
            self.get_label(labels, TaskType.INSTANCE_SEGMENTATION)
            if self.is_segmentation
            else None
        )

        output_nms_bboxes = self.get_input_tensors(inputs, "boundingbox")
        output_nms_masks = (
            self.get_input_tensors(inputs, "instance_segmentation")
            if self.is_segmentation
            else None
        )
        image_size = self.original_in_shape[1:]

        output_list: list[dict[str, Tensor]] = []
        label_list: list[dict[str, Tensor]] = []
        for i in range(len(output_nms_bboxes)):
            # Prepare predictions
            pred = {
                "boxes": output_nms_bboxes[i][:, :4],
                "scores": output_nms_bboxes[i][:, 4],
                "labels": output_nms_bboxes[i][:, 5].int(),
            }
            if self.is_segmentation:
                pred["masks"] = output_nms_masks[i].to(
                    dtype=torch.bool
                )  # Predicted masks (M, H, W)
            output_list.append(pred)

            # Prepare ground truth
            curr_label = box_label[box_label[:, 0] == i]
            curr_bboxs = box_convert(curr_label[:, 2:], "xywh", "xyxy")
            curr_bboxs[:, 0::2] *= image_size[1]
            curr_bboxs[:, 1::2] *= image_size[0]

            gt = {
                "boxes": curr_bboxs,
                "labels": curr_label[:, 1].int(),
            }
            if self.is_segmentation:
                gt["masks"] = mask_label[box_label[:, 0] == i].to(
                    dtype=torch.bool
                )
            label_list.append(gt)

        return output_list, label_list

    def reset(self) -> None:
        self.metric.reset()

    def compute(self) -> tuple[Tensor, dict[str, Tensor]]:
        metric_dict: dict[str, Tensor] = self.metric.compute()

        if self.is_segmentation:
            keys_to_remove = [
                "classes",
                "bbox_map_per_class",
                "bbox_mar_100_per_class",
                "segm_map_per_class",
                "segm_mar_100_per_class",
            ]
            for key in keys_to_remove:
                if key in metric_dict:
                    del metric_dict[key]

            for key in list(metric_dict.keys()):
                if "map" in key:
                    map_metric = metric_dict[key]
                    mar_key = key.replace("map", "mar")
                    if mar_key in metric_dict:
                        mar_metric = metric_dict[mar_key]
                        metric_dict[key.replace("map", "f1")] = (
                            2
                            * (map_metric * mar_metric)
                            / (map_metric + mar_metric)
                        )

            scalar = metric_dict.get("segm_map", torch.tensor(0.0))
        else:
            del metric_dict["classes"]
            del metric_dict["map_per_class"]
            del metric_dict["mar_100_per_class"]

            for key in list(metric_dict.keys()):
                if "map" in key:
                    map_metric = metric_dict[key]
                    mar_key = key.replace("map", "mar")
                    if mar_key in metric_dict:
                        mar_metric = metric_dict[mar_key]
                        metric_dict[key.replace("map", "f1")] = (
                            2
                            * (map_metric * mar_metric)
                            / (map_metric + mar_metric)
                        )

            scalar = metric_dict.pop("map", torch.tensor(0.0))

        return scalar, metric_dict
