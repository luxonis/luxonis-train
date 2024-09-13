import logging
from typing import Any

import torchmetrics
from luxonis_ml.data import LabelType
from torch import Tensor

from .base_metric import BaseMetric

logger = logging.getLogger(__name__)


class TorchMetricWrapper(BaseMetric[Tensor]):
    Metric: type[torchmetrics.Metric]

    def __init__(self, **kwargs: Any):
        super().__init__(node=kwargs.pop("node", None))
        task = kwargs.get("task")

        if self.n_classes > 1:
            if task == "binary":
                raise ValueError(
                    f"Task type set to '{task}', but the dataset has more than 1 class. "
                    f"Set the `task` parameter for {self.name} to either 'multiclass' or 'multilabel'."
                )
            task = "multiclass"
        else:
            if task == "multiclass":
                raise ValueError(
                    f"Task type set to '{task}', but the dataset has only 1 class. "
                    f"Set the `task` parameter for {self.name} to 'binary'."
                )
            task = "binary"
        if "task" not in kwargs:
            logger.warning(
                f"Task type not specified for {self.name}, assuming '{task}'. "
                "If this is not correct, please set the `task` parameter explicitly."
            )
        kwargs["task"] = task
        self._task = task

        if self._task == "multiclass":
            if "num_classes" not in kwargs:
                try:
                    kwargs["num_classes"] = self.n_classes
                except RuntimeError as e:
                    raise ValueError(
                        "Either `node` or `num_classes` must be provided to "
                        "multiclass torchmetrics."
                    ) from e
        else:
            if "num_labels" not in kwargs:
                try:
                    kwargs["num_labels"] = self.n_classes
                except RuntimeError as e:
                    raise ValueError(
                        "Either `node` or `num_labels` must be provided to "
                        "multilabel torchmetrics."
                    ) from e

        self.metric = self.Metric(**kwargs)

    def update(self, preds: Tensor, target: Tensor) -> None:
        if self._task in ["multiclass"]:
            target = target.argmax(dim=1)
        self.metric.update(preds, target)

    def compute(self) -> Tensor:
        return self.metric.compute()

    def reset(self) -> None:
        self.metric.reset()


class Accuracy(TorchMetricWrapper):
    supported_labels = [LabelType.CLASSIFICATION, LabelType.SEGMENTATION]
    Metric = torchmetrics.Accuracy


class F1Score(TorchMetricWrapper):
    supported_labels = [LabelType.CLASSIFICATION, LabelType.SEGMENTATION]
    Metric = torchmetrics.F1Score


class JaccardIndex(TorchMetricWrapper):
    supported_labels = [LabelType.CLASSIFICATION, LabelType.SEGMENTATION]
    Metric = torchmetrics.JaccardIndex


class Precision(TorchMetricWrapper):
    supported_labels = [LabelType.CLASSIFICATION, LabelType.SEGMENTATION]
    Metric = torchmetrics.Precision


class Recall(TorchMetricWrapper):
    supported_labels = [LabelType.CLASSIFICATION, LabelType.SEGMENTATION]
    Metric = torchmetrics.Recall
