from contextlib import suppress
from functools import cached_property

import torchmetrics
from loguru import logger
from torch import Tensor
from typing_extensions import override

from luxonis_train.tasks import Metadata, Tasks

from .base_metric import BaseMetric


class TorchMetricWrapper(BaseMetric):
    Metric: type[torchmetrics.Metric]

    def __init__(self, **kwargs):
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
            if task is not None:
                logger.warning(
                    "Parameter 'task' was not specified for `TorchMetric` "
                    f"based '{self.name}'. Assuming task type '{task}' "
                    "based on the number of classes. "
                    "If this is incorrect, please specify the "
                    "'task' parameter in the config."
                )

        if task is None:
            raise ValueError(
                f"'{self.name}' does not have the 'task' parameter set. "
                "and it is not possible to infer it from the other arguments. "
                "You can either set the 'task' parameter explicitly, "
                "provide either 'num_classes' or 'num_labels' argument, "
                "or use this metric with a node. "
                "The 'task' can be one of 'binary', 'multiclass', "
                "or 'multilabel'. "
            )
        if task not in {"binary", "multiclass", "multilabel"}:
            raise ValueError(
                f"Invalid task type '{task}' for '{self.name}'. "
                "The 'task' can be one of 'binary', 'multiclass', "
                "or 'multilabel'."
            )
        self._torchmetric_task = task
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
        if task != "binary" and n_classes == 1:
            raise ValueError(
                f"Task type set to '{task}', but the dataset has only 1 class. "
                f"Set the `task` argument of '{self.name}' to 'binary'."
            )

        if task == "multiclass":
            kwargs["num_classes"] = n_classes
        elif task == "multilabel":
            kwargs["num_labels"] = n_classes

        self.metric = self.Metric(**kwargs)

    @override
    def update(self, predictions: Tensor, target: Tensor) -> None:
        if self._torchmetric_task in ["multiclass"]:
            target = target.argmax(dim=1)
        self.metric.update(predictions, target)

    @override
    def compute(self) -> Tensor:
        x = self.metric.compute()
        if not (isinstance(x, Tensor) and x.ndim > 0 and x.numel() > 1):
            return x
        if getattr(self, "classes", None):
            metric_name = type(self.metric).__name__
            class_names = {v: k for k, v in self.classes.items()}
            return x.mean(), {
                metric_name + "_" + class_names[i]: x[i]
                for i in range(x.numel())
            }
        else:
            ValueError(
                f"Metric '{self.name}' does not have 'classes' attribute set."
            )

    @override
    def reset(self) -> None:
        self.metric.reset()

    @cached_property
    @override
    def required_labels(self) -> set[str | Metadata]:
        if self.task == Tasks.ANOMALY_DETECTION:
            return Tasks.SEGMENTATION.required_labels
        return self.task.required_labels


class Accuracy(TorchMetricWrapper):
    supported_tasks = [
        Tasks.CLASSIFICATION,
        Tasks.SEGMENTATION,
        Tasks.ANOMALY_DETECTION,
    ]
    Metric = torchmetrics.Accuracy


class F1Score(TorchMetricWrapper):
    supported_tasks = [
        Tasks.CLASSIFICATION,
        Tasks.SEGMENTATION,
        Tasks.ANOMALY_DETECTION,
    ]
    Metric = torchmetrics.F1Score


class JaccardIndex(TorchMetricWrapper):
    supported_tasks = [
        Tasks.CLASSIFICATION,
        Tasks.SEGMENTATION,
        Tasks.ANOMALY_DETECTION,
    ]
    Metric = torchmetrics.JaccardIndex


class Precision(TorchMetricWrapper):
    supported_tasks = [
        Tasks.CLASSIFICATION,
        Tasks.SEGMENTATION,
        Tasks.ANOMALY_DETECTION,
    ]
    Metric = torchmetrics.Precision


class Recall(TorchMetricWrapper):
    supported_tasks = [
        Tasks.CLASSIFICATION,
        Tasks.SEGMENTATION,
        Tasks.ANOMALY_DETECTION,
    ]
    Metric = torchmetrics.Recall
