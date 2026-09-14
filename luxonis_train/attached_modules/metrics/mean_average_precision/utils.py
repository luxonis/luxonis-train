"""Helpers of the mean average precision metrics.

`compute_metric_lists` converts a batch to the input of
``torchmetrics``. `postprocess_metrics` turns the raw results into the
main value and the other values that the trainer logs.

"""

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
    """Split the raw metric values into the main value and the others.

    The function adds the F1 scores with `add_f1_metrics`. It then
    splits each per-class tensor with `process_class_metrics`. At the
    end, it removes ``main_metric`` from the dictionary and returns that
    value separately. The function changes ``metrics`` in place.

    Args:
        metrics (``dict[str, Tensor]``): The raw metric values. The
            dictionary must hold the key ``"classes"``, see
            `process_class_metrics`.
        class_names (``Mapping[int, str]``): The class names, keyed by
            the class index.
        main_metric (str): The key of the main value, such as ``"map"``.
        device (torch.device): The device of the zero tensor that the
            function returns when ``metrics`` has no key
            ``main_metric``.

    Returns:
        ``tuple[Tensor, dict[str, Tensor]]``: The main value and the
        dictionary of the other values. The main value is a zero tensor
        on ``device`` when ``metrics`` has no key ``main_metric``.

    Example:
        >>> import torch
        >>> metrics = {
        ...     "map": torch.tensor(0.5),
        ...     "map_small": torch.tensor(0.5),
        ...     "mar_small": torch.tensor(1.0),
        ...     "classes": torch.tensor([0]),
        ... }
        >>> main, others = postprocess_metrics(
        ...     metrics, {0: "person"}, "map", torch.device("cpu")
        ... )
        >>> main.item()
        0.5
        >>> {key: round(value.item(), 3) for key, value in others.items()}
        {'map_small': 0.5, 'mar_small': 1.0, 'f1_small': 0.667}

    """
    metrics = process_class_metrics(add_f1_metrics(metrics), class_names)
    main_metric_value = metrics.pop(
        main_metric, torch.tensor(0.0, device=device)
    )
    return main_metric_value, metrics


def add_f1_metrics(metrics: dict[str, Tensor]) -> dict[str, Tensor]:
    r"""Add an F1 score for each pair of precision and recall values.

    For each key that contains ``"map"``, the function looks for the
    same key with ``"mar"`` in place of ``"map"``. When that key
    exists, the function adds the harmonic mean of the two values. The
    new key has ``"f1"`` in place of ``"map"``:

    .. math::

        F_1 = \frac{2 \cdot \text{mAP} \cdot \text{mAR}}{\text{mAP} + \text{mAR}}

    The F1 score is ``NaN`` when both values are ``0``. The function
    changes ``metrics`` in place.

    Args:
        metrics (``dict[str, Tensor]``): The metric values.

    Returns:
        ``dict[str, Tensor]``: The same dictionary, with the F1 scores
        added.

    Example:
        The key ``"mar"`` does not exist, so ``"map"`` gets no F1 score.

        >>> import torch
        >>> metrics = {
        ...     "map": torch.tensor(0.4),
        ...     "map_small": torch.tensor(0.5),
        ...     "mar_small": torch.tensor(1.0),
        ... }
        >>> result = add_f1_metrics(metrics)
        >>> sorted(result)
        ['f1_small', 'map', 'map_small', 'mar_small']
        >>> round(result["f1_small"].item(), 4)
        0.6667

    """
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
    """Split each per-class tensor into one scalar value per class.

    The function removes the key ``"classes"`` and every key that ends
    with ``"_per_class"``. A per-class tensor with more than one element
    gives one new key for each class, ``"<key>_<class name>"``. Its
    value is the element at the position of the class in
    ``"classes"``. A space in a class name becomes an underscore.

    A per-class tensor with one element gives no new keys. For example,
    ``torchmetrics`` returns ``[-1]`` when the per-class metrics are
    off. The function changes ``metrics`` in place.

    Args:
        metrics (``dict[str, Tensor]``): The metric values. The
            dictionary must hold the key ``"classes"``, a tensor with
            the class indices in the order of the per-class values.
        class_names (``Mapping[int, str]``): The class names, keyed by
            the class index. It must hold each index of ``"classes"``.

    Returns:
        ``dict[str, Tensor]``: The same dictionary, without
        ``"classes"`` and the per-class tensors, and with the value of
        each class.

    Example:
        >>> import torch
        >>> metrics = {
        ...     "map": torch.tensor(0.5),
        ...     "map_per_class": torch.tensor([0.25, 0.75]),
        ...     "classes": torch.tensor([0, 1]),
        ... }
        >>> names = {0: "person", 1: "traffic light"}
        >>> result = process_class_metrics(metrics, names)
        >>> sorted(result)
        ['map', 'map_per_class_person', 'map_per_class_traffic_light']
        >>> result["map_per_class_traffic_light"].item()
        0.75

    """
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
    """Convert a batch of boxes to the input of ``torchmetrics``.

    The function builds one prediction dictionary and one target
    dictionary for each image of the batch. Their format is the input
    of the ``update`` method of the ``torchmetrics``
    ``MeanAveragePrecision``:

    - A prediction dictionary holds ``"boxes"``, the first four columns
      of the predicted boxes. It also holds ``"scores"``, the fifth
      column, and ``"labels"``, the sixth column as ``int32``.
    - A target dictionary holds ``"boxes"``, the target boxes of the
      image in the ``xyxy`` format and in pixels. It also holds
      ``"labels"``, the class column as ``int32``.
    - With masks, each dictionary also holds ``"masks"``, the masks of
      the image as ``bool``.

    Args:
        boundinbox (``list[Tensor]``): The predicted boxes of each image,
            of shape ``[M_i, 6]``, as ``[x1, y1, x2, y2, score, class]``
            in pixels.
        target_boundingbox (``Tensor``): The target boxes of the batch,
            of shape ``[N, 6]``, as ``[batch_index, class, x, y, w, h]``.
            ``x`` and ``y`` are the normalized top-left corner, and
            ``w`` and ``h`` are the normalized size.
        height (int): The image height in pixels. It scales the ``y``
            coordinates of the target boxes.
        width (int): The image width in pixels. It scales the ``x``
            coordinates of the target boxes.
        masks (``list[Tensor] | None``): The predicted masks of each
            image, of shape ``[M_i, H, W]``. ``None`` adds no masks.
        target_masks (``Tensor | None``): The target masks of the batch,
            of shape ``[N, H, W]``, one for each row of
            ``target_boundingbox``. ``None`` adds no masks.

    Returns:
        ``tuple[list[dict[str, Tensor]], list[dict[str, Tensor]]]``:
        The prediction dictionaries and the target dictionaries, one of
        each for each item of ``boundinbox``.

    Raises:
        ValueError: When only one of ``masks`` and ``target_masks`` is
            given.

    Example:
        >>> import torch
        >>> boxes = [torch.tensor([[10.0, 20.0, 50.0, 60.0, 0.5, 1.0]])]
        >>> targets = torch.tensor([[0.0, 1.0, 0.25, 0.5, 0.5, 0.25]])
        >>> preds, target = compute_metric_lists(
        ...     boxes, targets, height=200, width=100
        ... )
        >>> preds[0]["scores"].tolist(), preds[0]["labels"].tolist()
        ([0.5], [1])
        >>> target[0]["boxes"].tolist()
        [[25.0, 100.0, 75.0, 150.0]]

    """
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
