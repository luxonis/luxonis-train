import torch
from torch import Tensor

from luxonis_train.nodes import OCRCTCHead
from luxonis_train.tasks import Tasks

from .base_metric import BaseMetric


class OCRAccuracy(BaseMetric):
    """Accuracy metric for OCR tasks."""

    supported_tasks = [Tasks.OCR]

    node: OCRCTCHead

    def __init__(self, blank_cls: int = 0, **kwargs):
        """Initializes the OCR accuracy metric.

        @type blank_cls: int
        @param blank_cls: Index of the blank class. Defaults to C{0}.
        """
        super().__init__(**kwargs)
        self.blank_cls = blank_cls
        self._init_metric()

    def _init_metric(self) -> None:
        """Initializes the running metric."""
        self.running_metric = {
            "acc_0": 0.0,
            "acc_1": 0.0,
            "acc_2": 0.0,
        }
        self.n = 0

    def update(self, predictions: Tensor, target: Tensor) -> None:
        """Updates the running metric with the given predictions and
        targets.

        @type preds: Tensor
        @param preds: A tensor containing the network predictions.
        @type targets: Tensor
        @param targets: A tensor containing the target labels.
        """

        target = self.node.encoder(target).to(predictions.device)

        B, T, _ = predictions.shape

        pred_classes = predictions.argmax(dim=-1)

        predictions = torch.zeros(
            (B, T), dtype=torch.int64, device=predictions.device
        )
        for i in range(B):
            unique_cons_classes = torch.unique_consecutive(pred_classes[i])
            unique_cons_classes = unique_cons_classes[
                unique_cons_classes != self.blank_cls
            ]
            if len(unique_cons_classes) != 0:
                predictions[i, : unique_cons_classes.shape[0]] = (
                    unique_cons_classes
                )

        target = torch.nn.functional.pad(
            target, (0, T - target.shape[1]), value=self.blank_cls
        )
        errors = predictions != target
        errors = errors.sum(dim=1)

        for acc_at in range(3):
            matching = (errors == acc_at) * 1.0
            self.running_metric[f"acc_{acc_at}"] += matching.sum().item()
        self.n += B

    def compute(self) -> tuple[Tensor, dict[str, Tensor]]:
        """Computes the OCR accuracy.

        @rtype: tuple[Tensor, dict[str, Tensor]]
        @return: A tuple containing the OCR accuracy and a dictionary of
            individual accuracies.
        """
        result = {
            "acc_0": torch.tensor(self.running_metric["acc_0"] / self.n),
            "acc_1": torch.tensor(self.running_metric["acc_1"] / self.n),
            "acc_2": torch.tensor(self.running_metric["acc_2"] / self.n),
        }
        return result["acc_0"], result

    def reset(self) -> None:
        self._init_metric()
