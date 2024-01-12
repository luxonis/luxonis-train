import logging

import torchmetrics

from .base_metric import BaseMetric

logger = logging.getLogger(__name__)


class TorchMetricWrapper(BaseMetric):
    def __init__(self, **kwargs):
        super().__init__(
            node=kwargs.pop("node", None),
            protocol=kwargs.pop("protocol", None),
            required_labels=kwargs.pop("required_labels", None),
        )
        task = kwargs.get("task")

        if task is None:
            if self.node.n_classes > 1:
                task = "multiclass"
            else:
                task = "binary"
            logger.warning(
                f"Task type not specified for {self.__class__.__name__}, "
                f"assuming {task}."
            )
            kwargs["task"] = task
        self.task = task

        if self.task == "multiclass":
            if "num_classes" not in kwargs:
                if self.node is None:
                    raise ValueError(
                        "Either `node` or `num_classes` must be provided to "
                        "multiclass torchmetrics."
                    )
                kwargs["num_classes"] = self.node.n_classes
        elif self.task == "multilabel":
            if "num_labels" not in kwargs:
                if self.node is None:
                    raise ValueError(
                        "Either `node` or `num_labels` must be provided to "
                        "multilabel torchmetrics."
                    )
                kwargs["num_labels"] = self.node.n_classes

        self.metric = self.Metric(**kwargs)

    def update(self, preds, target, *args, **kwargs):
        if self.task in ["multiclass"]:
            target = target.argmax(dim=1)
        self.metric.update(preds, target, *args, **kwargs)

    def compute(self):
        return self.metric.compute()


class Accuracy(TorchMetricWrapper):
    Metric = torchmetrics.Accuracy


class F1Score(TorchMetricWrapper):
    Metric = torchmetrics.F1Score


class JaccardIndex(TorchMetricWrapper):
    Metric = torchmetrics.JaccardIndex


class Precision(TorchMetricWrapper):
    Metric = torchmetrics.Precision


class Recall(TorchMetricWrapper):
    Metric = torchmetrics.Recall
