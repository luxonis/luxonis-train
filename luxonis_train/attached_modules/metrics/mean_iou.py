from typing import Literal

import torch
from torch import Tensor
from torchmetrics.segmentation import MeanIoU

from luxonis_train.tasks import Tasks

from .base_metric import BaseMetric


class MIoU(BaseMetric):
    """Mean Intersection over Union metric for segmentation masks.

    Metadata:
        - Module type: metric
        - Registry name: ``MIoU``
        - Task: SEGMENTATION
        - Attached node types: None
        - Inputs: ``predictions``, ``target``
        - Outputs: scalar mean IoU tensor, or scalar plus per-class IoU metrics
        - State: wrapped ``torchmetrics.segmentation.MeanIoU`` state

    Prediction format:
        ``predictions`` contains segmentation logits or one-hot masks, depending
        on ``input_format``.

    Target format:
        ``target`` contains segmentation labels in index or one-hot format,
        matching ``input_format``.

    Formula:
        Converts predictions and targets to the configured format and delegates
        IoU accumulation to ``torchmetrics.segmentation.MeanIoU``.

    Provenance:
        - Source: torchmetrics
        - License: Project license
        - Implementation notes: Can return per-class sub-metrics when
          ``per_class`` is enabled and class names are available from the node.

    """

    supported_tasks = [Tasks.SEGMENTATION]
    predefined_model_params_aliases = {"per_class_metrics": "per_class"}

    def __init__(
        self,
        num_classes: int,
        include_background: bool = True,
        per_class: bool = False,
        input_format: Literal["one-hot", "index"] = "index",
        **kwargs,
    ):
        """Initialize the mean IoU metric.

        Args:
            num_classes (int): Number of classes.
            include_background (bool): Whether to include the background class.
            per_class (bool): Whether to compute the IoU per class.
            input_format (``Literal["one-hot", "index"]``): Format of the input.
            **kwargs (``Any``): Keyword arguments forwarded to the parent class.

        """
        super().__init__(**kwargs)
        self.input_format = input_format
        self.include_background = include_background
        self.per_class = per_class
        self.metric = MeanIoU(
            num_classes=num_classes,
            include_background=include_background,
            per_class=per_class,
            input_format=input_format,
        )

    def convert_format(
        self, tensor: Tensor, is_target: bool = False
    ) -> Tensor:
        if self.input_format == "index":
            return torch.argmax(tensor, dim=1)
        if self.input_format == "one-hot" and not is_target:
            classes = torch.argmax(tensor, dim=1, keepdim=True)
            one_hot = torch.zeros_like(tensor)
            one_hot.scatter_(1, classes, 1)
            return one_hot
        return tensor

    def update(self, predictions: Tensor, target: Tensor) -> None:
        converted_preds = self.convert_format(predictions, is_target=False)

        if self.input_format == "index":
            converted_target = self.convert_format(target, is_target=True)
        elif self.input_format == "one-hot":
            converted_preds = converted_preds.bool()
            converted_target = target.bool()

        self.metric.update(converted_preds, converted_target)

    def compute(self) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        x = self.metric.compute()
        if not self.per_class or x.ndim == 0 or x.numel() == 1:
            return x

        class_names = [
            class_name
            for class_name, _ in sorted(
                self.classes.items(), key=lambda item: item[1]
            )
        ]
        if not self.include_background:
            class_names = class_names[1:]

        return x.mean(), {
            f"{self.name}_{class_name}": value
            for class_name, value in zip(class_names, x, strict=True)
        }

    def reset(self) -> None:
        self.metric.reset()
