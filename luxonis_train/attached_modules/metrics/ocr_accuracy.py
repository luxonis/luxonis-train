from typing import Annotated

import torch
import torch.nn.functional as F
from torch import Tensor

from luxonis_train.nodes import OCRCTCHead
from luxonis_train.tasks import Tasks

from .base_metric import BaseMetric, State


class OCRAccuracy(BaseMetric):
    """Accuracy metric for OCR tasks."""

    supported_tasks = [Tasks.OCR]

    node: OCRCTCHead

    acc_0: Annotated[
        Tensor, State(default=torch.tensor(0.0), dist_reduce_fx="sum")
    ]
    acc_1: Annotated[
        Tensor, State(default=torch.tensor(0.0), dist_reduce_fx="sum")
    ]
    acc_2: Annotated[
        Tensor, State(default=torch.tensor(0.0), dist_reduce_fx="sum")
    ]
    total: Annotated[
        Tensor, State(default=torch.tensor(0.0), dist_reduce_fx="sum")
    ]

    def __init__(self, blank_cls: int = 0, **kwargs):
        """Initializes the OCR accuracy metric.

        @type blank_cls: int
        @param blank_cls: Index of the blank class. Defaults to C{0}.
        """
        super().__init__(**kwargs)
        self.blank_class = blank_cls

    def update(self, predictions: Tensor, target: Tensor) -> None:
        """Updates the running metric with the given predictions and
        targets.

        @type preds: Tensor
        @param preds: A tensor containing the network predictions.
        @type targets: Tensor
        @param targets: A tensor containing the target labels.
        """

        target = self.node.encoder(target).to(predictions.device)

        batch_size, text_length, _ = predictions.shape

        pred_classes = predictions.argmax(dim=-1)

        predictions = torch.zeros(
            (batch_size, text_length),
            dtype=torch.int64,
            device=predictions.device,
        )
        for i in range(batch_size):
            unique_cons_classes = torch.unique_consecutive(pred_classes[i])
            unique_cons_classes = unique_cons_classes[
                unique_cons_classes != self.blank_class
            ]
            if len(unique_cons_classes) != 0:
                predictions[i, : unique_cons_classes.shape[0]] = (
                    unique_cons_classes
                )

        target = F.pad(
            target, (0, text_length - target.shape[1]), value=self.blank_class
        )
        errors = (predictions != target).sum(dim=1)

        for acc_at in range(3):
            matching = (errors == acc_at) * 1.0
            [self.acc_0, self.acc_1, self.acc_2][acc_at] += matching.sum()
        self.total += batch_size

    def compute(self) -> tuple[Tensor, dict[str, Tensor]]:
        """Computes the OCR accuracy.

        @rtype: tuple[Tensor, dict[str, Tensor]]
        @return: A tuple containing the OCR accuracy and a dictionary of
            individual accuracies.
        """
        return self.acc_0 / self.total, {
            "acc_0": self.acc_0 / self.total,
            "acc_1": self.acc_1 / self.total,
            "acc_2": self.acc_2 / self.total,
        }
