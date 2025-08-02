from typing import Annotated

import torch
import torch.nn.functional as F
from torch import Tensor
from typing_extensions import override

from luxonis_train.nodes import OCRCTCHead
from luxonis_train.tasks import Tasks

from .base_metric import BaseMetric, MetricState


class OCRAccuracy(BaseMetric):
    """Accuracy metric for OCR tasks."""

    supported_tasks = [Tasks.OCR]

    node: OCRCTCHead

    rank_0: Annotated[Tensor, MetricState()]
    rank_1: Annotated[Tensor, MetricState()]
    rank_2: Annotated[Tensor, MetricState()]
    total: Annotated[Tensor, MetricState()]

    def __init__(self, blank_class: int = 0, **kwargs):
        """Initializes the OCR accuracy metric.

        @type blank_class: int
        @param blank_class: Index of the blank class. Defaults to C{0}.
        """
        super().__init__(**kwargs)
        self.blank_class = blank_class

    @override
    def update(self, predictions: Tensor, target: Tensor) -> None:
        """Updates the running metric with the given predictions and
        targets.

        @type predictions: Tensor
        @param predictions: A tensor containing the network predictions.
        @type targets: Tensor
        @param targets: A tensor containing the target labels.
        """
        target = self.node.encoder(target).to(self.device)

        batch_size, text_length, _ = predictions.shape

        pred_classes = predictions.argmax(dim=-1)

        predictions = torch.zeros(
            (batch_size, text_length), dtype=torch.int64, device=self.device
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

        for rank in range(3):
            matching = (errors == rank) * 1.0
            [self.rank_0, self.rank_1, self.rank_2][rank] += matching.sum()
        self.total += batch_size

    @override
    def compute(self) -> tuple[Tensor, dict[str, Tensor]]:
        """Computes the OCR accuracy.

        @rtype: tuple[Tensor, dict[str, Tensor]]
        @return: A tuple containing the OCR accuracy and a dictionary of
            individual accuracies.
        """
        return self.rank_0 / self.total, {
            "rank_0": self.rank_0 / self.total,
            "rank_1": self.rank_1 / self.total,
            "rank_2": self.rank_2 / self.total,
        }
