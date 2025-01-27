from functools import cached_property
from typing import Any

import torch
import torchmetrics.detection as detection
from torch import Tensor
from torchvision.ops import box_convert
from typing_extensions import override

from luxonis_train.enums import Task

from .base_metric import BaseMetric


class MeanAveragePrecision(BaseMetric):
    """Compute the Mean-Average-Precision (mAP) and Mean-Average-Recall
    (mAR) for object detection predictions and instance segmentation.

    Adapted from U{Mean-Average-Precision (mAP) and Mean-Average-Recall
    (mAR)
    <https://lightning.ai/docs/torchmetrics/stable/detection/mean_average_precision.html>}.
    """

    supported_tasks = [Task.BOUNDINGBOX, Task.INSTANCE_SEGMENTATION]

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.is_segmentation = self.node.task is Task.INSTANCE_SEGMENTATION

        if self.is_segmentation:
            iou_type = ("bbox", "segm")
        else:
            iou_type = "bbox"

        self.metric = detection.MeanAveragePrecision(iou_type=iou_type)  # type: ignore

    @cached_property
    @override
    def required_labels(self) -> set[str]:
        return Task.BOUNDINGBOX.required_labels

    def update(self, predictions: list[Tensor], targets: Tensor) -> None:
        image_size = self.original_in_shape[1:]

        output_list: list[dict[str, Tensor]] = []
        label_list: list[dict[str, Tensor]] = []
        for i in range(len(predictions)):
            # Prepare predictions
            pred = {
                "boxes": predictions[i][:, :4],
                "scores": predictions[i][:, 4],
                "labels": predictions[i][:, 5].int(),
            }
            if self.is_segmentation:
                pred["masks"] = predictions[i].to(  # type: ignore
                    dtype=torch.bool
                )  # Predicted masks (M, H, W)
            output_list.append(pred)

            # Prepare ground truth
            curr_label = targets[targets[:, 0] == i]
            curr_bboxs = box_convert(curr_label[:, 2:], "xywh", "xyxy")
            curr_bboxs[:, 0::2] *= image_size[1]
            curr_bboxs[:, 1::2] *= image_size[0]

            gt = {
                "boxes": curr_bboxs,
                "labels": curr_label[:, 1].int(),
            }
            if self.is_segmentation:
                gt["masks"] = targets[targets[:, 0] == i].to(  # type: ignore
                    dtype=torch.bool
                )
            label_list.append(gt)

        self.metric.update(output_list, label_list)

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

        # WARNING: fix DDP pl.log error
        metric_dict = {k: v.to(self.device) for k, v in metric_dict.items()}
        scalar = scalar.to(self.device)

        return scalar, metric_dict
