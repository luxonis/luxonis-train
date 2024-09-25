from typing import Any

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
    (mAR) for object detection predictions.

    Adapted from U{Mean-Average-Precision (mAP) and Mean-Average-Recall
    (mAR)
    <https://lightning.ai/docs/torchmetrics/stable/detection/mean_average_precision.html>}.
    """

    supported_tasks: list[TaskType] = [TaskType.BOUNDINGBOX]

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.metric = detection.MeanAveragePrecision()

    def update(
        self,
        outputs: list[dict[str, Tensor]],
        labels: list[dict[str, Tensor]],
    ):
        self.metric.update(outputs, labels)

    def prepare(
        self, inputs: Packet[Tensor], labels: Labels
    ) -> tuple[list[dict[str, Tensor]], list[dict[str, Tensor]]]:
        box_label = self.get_label(labels)
        output_nms = self.get_input_tensors(inputs)

        image_size = self.original_in_shape[1:]

        output_list: list[dict[str, Tensor]] = []
        label_list: list[dict[str, Tensor]] = []
        for i in range(len(output_nms)):
            output_list.append(
                {
                    "boxes": output_nms[i][:, :4],
                    "scores": output_nms[i][:, 4],
                    "labels": output_nms[i][:, 5].int(),
                }
            )

            curr_label = box_label[box_label[:, 0] == i]
            curr_bboxs = box_convert(curr_label[:, 2:], "xywh", "xyxy")
            curr_bboxs[:, 0::2] *= image_size[1]
            curr_bboxs[:, 1::2] *= image_size[0]
            label_list.append(
                {"boxes": curr_bboxs, "labels": curr_label[:, 1].int()}
            )

        return output_list, label_list

    def reset(self) -> None:
        self.metric.reset()

    def compute(self) -> tuple[Tensor, dict[str, Tensor]]:
        metric_dict: dict[str, Tensor] = self.metric.compute()

        del metric_dict["classes"]
        del metric_dict["map_per_class"]
        del metric_dict["mar_100_per_class"]
        for key in list(metric_dict.keys()):
            if "map" in key:
                map = metric_dict[key]
                mar_key = key.replace("map", "mar")
                if mar_key in metric_dict:
                    mar = metric_dict[mar_key]
                    metric_dict[key.replace("map", "f1")] = (
                        2 * (map * mar) / (map + mar)
                    )

        map = metric_dict.pop("map")

        return map, metric_dict
