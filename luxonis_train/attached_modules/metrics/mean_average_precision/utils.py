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
            map = metrics[key]
            mar_key = key.replace("map", "mar")
            if mar_key in metrics:
                mar = metrics[mar_key]
                metrics[key.replace("map", "f1")] = (
                    2 * (map * mar) / (map + mar)
                )
    return metrics


def process_class_metrics(
    metrics: dict[str, Tensor], class_names: Mapping[int, str]
) -> dict[str, Tensor]:
    classes = metrics.pop("classes")
    per_class_metrics = [key for key in metrics if key.endswith("_per_class")]

    for metric_name in per_class_metrics:
        metric = metrics.pop(metric_name)

        if metric.shape[0] > 1:
            for i, class_id in enumerate(classes):
                class_name = class_names[int(class_id)].replace(" ", "_")
                metrics[f"{metric_name}_{class_name}"] = metric[i]

    return metrics


def compute_metric_lists(
    boundinbox: list[Tensor],
    target_boundingbox: Tensor,
    height: int,
    width: int,
    *,
    masks: list[Tensor] | None = None,
    target_masks: Tensor | None = None,
) -> tuple[list[dict[str, Tensor]], list[dict[str, Tensor]]]:
    if any_not_none([masks, target_masks]) and not all_not_none(
        [masks, target_masks]
    ):
        raise ValueError(
            "Either both `masks` and `target_masks` "
            "must be provided, or neither."
        )
    predictions: list[dict[str, Tensor]] = []
    targets: list[dict[str, Tensor]] = []
    for i in range(len(boundinbox)):
        pred = {
            "boxes": boundinbox[i][:, :4],
            "scores": boundinbox[i][:, 4],
            "labels": boundinbox[i][:, 5].int(),
        }

        bboxes_target = target_boundingbox[target_boundingbox[:, 0] == i]
        bboxes = box_convert(bboxes_target[:, 2:6], "xywh", "xyxy")
        bboxes[:, 0::2] *= width
        bboxes[:, 1::2] *= height

        target = {"boxes": bboxes, "labels": bboxes_target[:, 1].int()}

        if masks is not None and target_masks is not None:
            pred["masks"] = masks[i].bool()
            target["masks"] = target_masks[
                target_boundingbox[:, 0] == i
            ].bool()

        targets.append(target)
        predictions.append(pred)

    return predictions, targets
