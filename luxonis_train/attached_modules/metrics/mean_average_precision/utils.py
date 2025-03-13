from collections.abc import Mapping

import torch
from luxonis_ml.typing import all_not_none, any_not_none
from torch import Tensor
from torchvision.ops import box_convert


def postprocess_metrics(
    metrics: dict[str, Tensor],
    class_names: Mapping[int, str],
    main_metric: str,
    device: torch.device,
) -> tuple[Tensor, dict[str, Tensor]]:
    metrics = process_class_metrics(add_f1_metrics(metrics), class_names)
    main_metric_value = metrics.pop(
        main_metric, torch.tensor(0.0, device=device)
    )
    return main_metric_value, metrics


def add_f1_metrics(metrics: dict[str, Tensor]) -> dict[str, Tensor]:
    for key in list(metrics.keys()):
        if "map" in key:
            map_metric = metrics[key]
            mar_key = key.replace("map", "mar")
            if mar_key in metrics:
                mar_metric = metrics[mar_key]
                metrics[key.replace("map", "f1")] = (
                    2 * (map_metric * mar_metric) / (map_metric + mar_metric)
                )
    return metrics


def process_class_metrics(
    metrics: dict[str, Tensor], class_names: Mapping[int, str]
) -> dict[str, Tensor]:
    classes = metrics.pop("classes")
    per_class_metrics = [key for key in metrics if key.endswith("_per_class")]

    for metric_name in per_class_metrics:
        metric = metrics.pop(metric_name)

        if metric.ndim > 1:
            for i, class_id in enumerate(classes):
                class_name = class_names[int(class_id)].replace(" ", "_")
                metrics[f"{metric_name}_{class_name}"] = metric[i]

    return metrics


def compute_update_lists(
    boundinbox: list[Tensor],
    targets: Tensor,
    height: int,
    width: int,
    *,
    keypoints: list[Tensor] | None = None,
    n_keypoints: int | None = None,
) -> tuple[list[dict[str, Tensor]], list[dict[str, Tensor]]]:
    if any_not_none([keypoints, n_keypoints]) and not all_not_none(
        [keypoints, n_keypoints]
    ):
        raise ValueError(
            "Either both `keypoints` and `n_keypoints` "
            "must be provided, or neither."
        )
    output_list: list[dict[str, Tensor]] = []
    label_list: list[dict[str, Tensor]] = []
    for i in range(len(boundinbox)):
        pred = {
            "boxes": boundinbox[i][:, :4],
            "scores": boundinbox[i][:, 4],
            "labels": boundinbox[i][:, 5].int(),
        }
        if keypoints is not None and n_keypoints is not None:
            pred["keypoints"] = keypoints[i].reshape(-1, n_keypoints * 3)

        output_list.append(pred)

        target = targets[targets[:, 0] == i]
        bboxs = box_convert(target[:, 2:6], "xywh", "xyxy")
        bboxs[:, 0::2] *= width
        bboxs[:, 1::2] *= height

        gt = {
            "boxes": bboxs,
            "labels": target[:, 1].int(),
        }
        label_list.append(gt)

    return output_list, label_list
