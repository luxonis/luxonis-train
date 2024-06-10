import logging

import torch
from .base_metric import BaseMetric

logger = logging.getLogger(__name__)


class OCRAccuracy(BaseMetric):
    def __init__(self, **kwargs):
        super().__init__(
            node=kwargs.pop("node", None),
            protocol=kwargs.pop("protocol", None),
            required_labels=kwargs.pop("required_labels", None),
        )
        self.blank_cls = kwargs.get("task")
        self._init_metric()

    def _init_metric(self):
        self.running_metric = {
            "acc_0": 0,
            "acc_1": 0,
            "acc_2": 0
        }
        self.n = 0

    def update(self, preds, target, *args, **kwargs):
        B, C, T = preds.shape  # batch, class, step
        target, _, _ = target
        preds = preds.softmax(dim=1)
        pred_classes = preds.argmax(dim=1)  # batch, step
        pred_classes = torch.unique_consecutive(pred_classes, dim=1)
        pred_classes_aligned = torch.zeros_like(pred_classes)
        for idx, pred_cls in enumerate(pred_classes):
            aligned_cls = [cls for cls in pred_classes if len(cls) > self.blank_cls]
            aligned_cls = aligned_cls + [0 for _ in range(T - len(aligned_cls))]
            pred_classes_aligned[idx] = torch.tensor(aligned_cls).to(pred_classes.device)

        errors = pred_classes_aligned == target
        errors = errors.sum(dim=1)

        for acc_at in range(3):
            matching = (errors == acc_at) * 1.0
            self.running_metric[f"acc_{acc_at}"] += matching.sum().item()
        self.n += B

    def compute(self):
        result = {
            "acc_0": self.running_metric["acc_0"] / self.n,
            "acc_1": self.running_metric["acc_1"] / self.n,
            "acc_2": self.running_metric["acc_2"] / self.n
        }
        self._init_metric()
        return result["acc_0"], result
