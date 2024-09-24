import logging
from contextlib import suppress
from typing import Any

import torchmetrics
from torch import Tensor

from luxonis_train.enums import TaskType

from .base_metric import BaseMetric

logger = logging.getLogger(__name__)


class TorchMetricWrapper(BaseMetric[Tensor]):
    Metric: type[torchmetrics.Metric]

    def __init__(self, **kwargs: Any):
        super().__init__(node=kwargs.pop("node", None))
        task = kwargs.get("task")
        if task is None:
            if "num_classes" in kwargs:
                if kwargs["num_classes"] == 1:
                    task = "binary"
                else:
                    task = "multiclass"
            elif "num_labels" in kwargs:
                task = "multilabel"
            else:
                with suppress(RuntimeError, ValueError):
                    if self.n_classes == 1:
                        task = "binary"
                    else:
                        task = "multiclass"

        if task is None:
            raise ValueError(
                f"'{self.name}' does not have the 'task' parameter set. "
                "and it is not possible to infer it from the other arguments. "
                "You can either set the 'task' parameter explicitly, provide either 'num_classes' or 'num_labels' argument, "
                "or use this metric with a node. "
                "The 'task' can be one of 'binary', 'multiclass', or 'multilabel'. "
            )
        self._task = task
        kwargs["task"] = task

        n_classes: int | None = kwargs.get(
            "num_classes", kwargs.get("num_labels")
        )

        if n_classes is None:
            with suppress(RuntimeError, ValueError):
                n_classes = self.n_classes

        if n_classes is None and task != "binary":
            arg_name = "num_classes" if task == "multiclass" else "num_labels"
            raise ValueError(
                f"'{self.name}' metric does not have the '{arg_name}' parameter set "
                "and it is not possible to infer it from the other arguments. "
                "You can either set the '{arg_name}' parameter explicitly, or use this metric with a node."
            )

        if task == "binary" and n_classes is not None and n_classes > 1:
            raise ValueError(
                f"Task type set to '{task}', but the dataset has more than 1 class. "
                f"Set the `task` argument of '{self.name}' to either 'multiclass' or 'multilabel'."
            )
        elif task != "binary" and n_classes == 1:
            raise ValueError(
                f"Task type set to '{task}', but the dataset has only 1 class. "
                f"Set the `task` argument of '{self.name}' to 'binary'."
            )

        if task == "multiclass":
            kwargs["num_classes"] = n_classes
        elif task == "multilabel":
            kwargs["num_labels"] = n_classes

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
    supported_tasks: list[TaskType] = [
        TaskType.CLASSIFICATION,
        TaskType.SEGMENTATION,
    ]
    Metric = torchmetrics.Accuracy


class F1Score(TorchMetricWrapper):
    supported_tasks: list[TaskType] = [
        TaskType.CLASSIFICATION,
        TaskType.SEGMENTATION,
    ]
    Metric = torchmetrics.F1Score


class JaccardIndex(TorchMetricWrapper):
    supported_tasks: list[TaskType] = [
        TaskType.CLASSIFICATION,
        TaskType.SEGMENTATION,
    ]
    Metric = torchmetrics.JaccardIndex


class Precision(TorchMetricWrapper):
    supported_tasks: list[TaskType] = [
        TaskType.CLASSIFICATION,
        TaskType.SEGMENTATION,
    ]
    Metric = torchmetrics.Precision


class Recall(TorchMetricWrapper):
    supported_tasks: list[TaskType] = [
        TaskType.CLASSIFICATION,
        TaskType.SEGMENTATION,
    ]
    Metric = torchmetrics.Recall
