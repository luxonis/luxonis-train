import torch
import torchmetrics.detection as detection
from torch import Tensor
from torchvision.ops import box_convert

from luxonis_train.attached_modules.metrics import BaseMetric
from luxonis_train.tasks import Tasks


class MeanAveragePrecisionBBox(BaseMetric, register=False):
    """Compute the Mean-Average-Precision (mAP) and Mean-Average-Recall
    (mAR) for object detection predictions and instance segmentation.

    Adapted from U{Mean-Average-Precision (mAP) and Mean-Average-Recall
    (mAR)
    <https://lightning.ai/docs/torchmetrics/stable/detection/mean_average_precision.html>}.
    """

    supported_tasks = [Tasks.BOUNDINGBOX]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.metric = detection.MeanAveragePrecision(iou_type="bbox")

    def compute_update_lists(
        self, predictions: list[Tensor], targets: Tensor
    ) -> tuple[list[dict[str, Tensor]], list[dict[str, Tensor]]]:
        image_size = self.original_in_shape[1:]

        output_list: list[dict[str, Tensor]] = []
        label_list: list[dict[str, Tensor]] = []
        for i in range(len(predictions)):
            pred = {
                "boxes": predictions[i][:, :4],
                "scores": predictions[i][:, 4],
                "labels": predictions[i][:, 5].int(),
            }
            output_list.append(pred)

            curr_label = targets[targets[:, 0] == i]
            curr_bboxs = box_convert(curr_label[:, 2:], "xywh", "xyxy")
            curr_bboxs[:, 0::2] *= image_size[1]
            curr_bboxs[:, 1::2] *= image_size[0]

            gt = {
                "boxes": curr_bboxs,
                "labels": curr_label[:, 1].int(),
            }
            label_list.append(gt)

        return output_list, label_list

    def update(self, predictions: list[Tensor], targets: Tensor) -> None:
        self.metric.update(*self.compute_update_lists(predictions, targets))

    def reset(self) -> None:
        self.metric.reset()

    def compute(self) -> tuple[Tensor, dict[str, Tensor]]:
        metric_dict: dict[str, Tensor] = self.metric.compute()

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
