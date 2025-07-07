from contextlib import suppress
from functools import cached_property

import torchmetrics
from torch import Tensor
from typing_extensions import override

from luxonis_train.tasks import Metadata, Tasks

from .base_metric import BaseMetric


class TorchMetricWrapper(BaseMetric):
    Metric: type[torchmetrics.Metric]

    def __init__(self, **kwargs):
        super().__init__(node=kwargs.pop("node", None))
        task = self._infer_torchmetrics_task(**kwargs)
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
        if self._torchmetric_task == "multiclass":
            target = target.argmax(dim=1)
        self.metric.update(predictions, target)

    @override
    def compute(self) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
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
        raise ValueError(
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
