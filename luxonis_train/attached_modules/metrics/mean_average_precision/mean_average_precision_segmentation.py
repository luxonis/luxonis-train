import torch
import torchmetrics.detection as detection
from torch import Tensor

from luxonis_train.tasks import Tasks

from .mean_average_precision_bbox import MeanAveragePrecisionBBox


class MeanAveragePrecisionSegmentation(
    MeanAveragePrecisionBBox, register=False
):
    supported_tasks = [Tasks.INSTANCE_SEGMENTATION]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.metric = detection.MeanAveragePrecision(
            iou_type=("bbox", "segm"),  # type: ignore
        )

    def update(
        self,
        boundingbox: list[Tensor],
        instance_segmentation: list[Tensor],
        target_boundingbox: Tensor,
        target_instance_segmentation: Tensor,
    ) -> None:
        output_list, label_list = self.compute_update_lists(
            boundingbox, target_boundingbox
        )

        for i in range(len(instance_segmentation)):
            output_list[i]["masks"] = instance_segmentation[i].to(
                dtype=torch.bool
            )

            label_list[i]["masks"] = target_instance_segmentation[
                target_boundingbox[:, 0] == i
            ].to(dtype=torch.bool)

        self.metric.update(output_list, label_list)

    def reset(self) -> None:
        self.metric.reset()

    def compute(self) -> tuple[Tensor, dict[str, Tensor]]:
        metric_dict: dict[str, Tensor] = self.metric.compute()

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

        # WARNING: fix DDP pl.log error
        metric_dict = {k: v.to(self.device) for k, v in metric_dict.items()}
        scalar = scalar.to(self.device)

        return scalar, metric_dict
