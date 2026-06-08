from typing import Literal

import torch
from torch import Tensor
from torchmetrics.segmentation import DiceScore

from luxonis_train.tasks import Tasks

from .base_metric import BaseMetric


class DiceCoefficient(BaseMetric):
    """Dice coefficient metric for segmentation masks.

    Metadata:
        - Module type: metric
        - Registry name: ``DiceCoefficient``
        - Task: SEGMENTATION
        - Attached node types: None
        - Inputs: ``predictions``, ``target``
        - Outputs: scalar Dice coefficient tensor
        - State: wrapped ``torchmetrics.segmentation.DiceScore`` state

    Prediction format:
        ``predictions`` contains segmentation logits or one-hot masks, depending
        on ``input_format``.

    Target format:
        ``target`` contains segmentation labels in index or one-hot format,
        matching ``input_format``.

    Formula:
        Converts predictions and targets to the configured format and delegates
        Dice computation to ``torchmetrics.segmentation.DiceScore``.

    Provenance:
        - Source: torchmetrics
        - License: Project license
        - Implementation notes: Converts logits to argmax labels or one-hot
          masks before updating the wrapped metric.

    """

    supported_tasks = [Tasks.SEGMENTATION]

    def __init__(
        self,
        num_classes: int,
        include_background: bool = True,
        average: Literal["micro", "macro", "weighted", "none"]
        | None = "micro",
        input_format: Literal["one-hot", "index"] = "index",
        **kwargs,
    ):
        """Initialize the Dice coefficient metric.

        Args:
            num_classes (int): Number of classes.
            include_background (bool): Whether to include the background class.
            average (``Literal["micro", "macro", "weighted", "none"]``): ``Type`` of averaging.
            input_format (``Literal["one-hot", "index"]``): Format of the input.
            **kwargs (``Any``): Keyword arguments forwarded to the parent class.

        """
        super().__init__(**kwargs)
        self.input_format = input_format
        self.metric = DiceScore(
            num_classes=num_classes,
            include_background=include_background,
            average=average,
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

    def compute(self) -> Tensor:
        return self.metric.compute()

    def reset(self) -> None:
        self.metric.reset()
