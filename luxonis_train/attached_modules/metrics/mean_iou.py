from typing import Literal

import torch
from torch import Tensor
from torchmetrics.segmentation import MeanIoU

from luxonis_train.tasks import Tasks

from .base_metric import BaseMetric


class MIoU(BaseMetric):
    """Mean IoU metric for SEGMENTATION tasks."""

    supported_tasks = [Tasks.SEGMENTATION]

    def __init__(
        self,
        num_classes: int,
        include_background: bool = True,
        per_class: bool = False,
        input_format: Literal["one-hot", "index"] = "index",
        **kwargs,
    ):
        """Initializes the mean IoU metric.

        @type num_classes: int
        @param num_classes: Number of classes.
        @type include_background: bool
        @param include_background: Whether to include the background
            class.
        @type per_class: bool
        @param per_class: Whether to compute the IoU per class.
        @type input_format: Literal["one-hot", "index"]
        @param input_format: Format of the input.
        """
        super().__init__(**kwargs)
        self.input_format = input_format
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

    def compute(self) -> Tensor:
        return self.metric.compute()

    def reset(self) -> None:
        self.metric.reset()
