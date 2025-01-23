import torch
from torch import Tensor
from .base_metric import BaseMetric
import logging

from luxonis_train.utils import (
    Labels,
    Packet,
    prepare_batch_targets
)

logger = logging.getLogger(__name__)


class OCRAccuracy(BaseMetric[list[dict[str, Tensor]], list[dict[str, Tensor]]]):
    """Accuracy metric for OCR tasks."""
    def __init__(self, blank_cls: int = 0, **kwargs):
        """Initializes the OCR accuracy metric.

        @type blank_cls: int
        @param blank_cls: Index of the blank class. Defaults to C{0}.
        """
        super().__init__(**kwargs)
        self.blank_cls = blank_cls
        self._init_metric()

    def _init_metric(self):
        """Initializes the running metric."""
        self.running_metric = {
            "acc_0": 0.0,
            "acc_1": 0.0,
            "acc_2": 0.0,
        }
        self.n = 0


    def prepare(
        self, inputs: Packet[Tensor], labels: Labels
    ) -> tuple[Tensor, Tensor]:
        """Prepares the predictions and targets for accuracy computation.

        @type inputs: Packet[Tensor]
        @param inputs: A packet containing input tensors, typically network predictions.
        @type labels: Labels
        @param labels: A dictionary containing text labels and corresponding lengths.
        @rtype: tuple[Tensor, Tensor]
        @return: A tuple of predictions and targets.
        """

        preds = inputs['/classification'][0]
        targets_batch = labels['/metadata/text']
        target_lengths = labels['/metadata/text_length'].int()

        targets = prepare_batch_targets(targets_batch, target_lengths)
        targets = self.node.encoder(targets).to(preds.device) # type: ignore

        return (preds, targets)
    
    def update(self, preds: Tensor, targets: Tensor) -> None:
        """Updates the running metric with the given predictions and targets.

        @type preds: Tensor
        @param preds: A tensor containing the network predictions.
        @type targets: Tensor
        @param targets: A tensor containing the target labels.
        """

        B, T, C = preds.shape 

        pred_classes = preds.argmax(dim=-1)

        preds = torch.zeros((B, T), dtype=torch.int64, device=preds.device)
        for i in range(B):
            unique_cons_classes = torch.unique_consecutive(pred_classes[i])
            unique_cons_classes = unique_cons_classes[unique_cons_classes != self.blank_cls]
            if len(unique_cons_classes) != 0:
                preds[i, :unique_cons_classes.shape[0]] = unique_cons_classes
    
        target = torch.nn.functional.pad(targets, (0, T - targets.shape[1]), value=self.blank_cls)
        errors = preds != target
        errors = errors.sum(dim=1)

        for acc_at in range(3):
            matching = (errors == acc_at) * 1.0
            self.running_metric[f"acc_{acc_at}"] += matching.sum().item()
        self.n += B

    def compute(self) -> tuple[Tensor, dict[str, Tensor]]:
        """Computes the OCR accuracy.

        @rtype: tuple[Tensor, dict[str, Tensor]]
        @return: A tuple containing the OCR accuracy and a dictionary of individual accuracies.
        """
        result = {
            "acc_0": torch.tensor(self.running_metric["acc_0"] / self.n),
            "acc_1": torch.tensor(self.running_metric["acc_1"] / self.n),
            "acc_2": torch.tensor(self.running_metric["acc_2"] / self.n),
        }
        self._init_metric()
        return result["acc_0"], result