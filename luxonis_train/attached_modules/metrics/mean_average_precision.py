import torchmetrics.detection as detection
from torch import Tensor
from torchvision.ops import box_convert

from luxonis_train.utils.types import (
    BBoxProtocol,
    Labels,
    LabelType,
    Packet,
)

from .base_metric import BaseMetric


class MeanAveragePrecision(BaseMetric, detection.MeanAveragePrecision):
    """Compute the Mean-Average-Precision (mAP) and Mean-Average-Recall (mAR) for object
    detection predictions.

    Adapted from U{Mean-Average-Precision (mAP) and Mean-Average-Recall (mAR)
    <https://lightning.ai/docs/torchmetrics/stable/detection/mean_average_precision.html>}.
    """

    def __init__(self, **kwargs):
        super().__init__(
            protocol=BBoxProtocol,
            required_labels=[LabelType.BOUNDINGBOX],
            **kwargs,
        )
        self.metric = detection.MeanAveragePrecision()

    def update(
        self,
        outputs: list[dict[str, Tensor]],
        labels: list[dict[str, Tensor]],
    ):
        self.metric.update(outputs, labels)

    def prepare(
        self, outputs: Packet[Tensor], labels: Labels
    ) -> tuple[list[dict[str, Tensor]], list[dict[str, Tensor]]]:
        label = labels[LabelType.BOUNDINGBOX]
        output_nms = outputs["boxes"]

        image_size = self.node.original_in_shape[2:]

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

            curr_label = label[label[:, 0] == i]
            curr_bboxs = box_convert(curr_label[:, 2:], "xywh", "xyxy")
            curr_bboxs[:, 0::2] *= image_size[1]
            curr_bboxs[:, 1::2] *= image_size[0]
            label_list.append({"boxes": curr_bboxs, "labels": curr_label[:, 1].int()})

        return output_list, label_list

    def compute(self) -> tuple[Tensor, dict[str, Tensor]]:
        metric_dict = self.metric.compute()

        del metric_dict["classes"]
        del metric_dict["map_per_class"]
        del metric_dict["mar_100_per_class"]
        map = metric_dict.pop("map")

        return map, metric_dict
