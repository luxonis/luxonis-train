from contextlib import suppress
from functools import cached_property

import torchmetrics
from torch import Tensor
from typing_extensions import override

from luxonis_train.tasks import Metadata, Tasks

from .base_metric import BaseMetric


class TorchMetricWrapper(BaseMetric):
    """Base wrapper for torchmetrics classification-style metrics.

    Metadata:
        - Module type: metric
        - Registry name: ``TorchMetricWrapper``
        - Task: None
        - Attached node types: None
        - Inputs: ``predictions``, ``target``
        - Outputs: scalar metric tensor, or scalar plus per-class sub-metrics
        - State: wrapped ``torchmetrics.Metric`` state

    Prediction format:
        ``predictions`` contains logits or scores accepted by the wrapped
        torchmetrics metric.

    Target format:
        ``target`` contains class labels or segmentation labels required by the
        inferred torchmetrics task.

    Provenance:
        - Source: torchmetrics
        - License: Apache License 2.0
        - Implementation notes: Infers binary, multiclass, or multilabel mode
          from parameters or the attached node.

    """

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
    """TorchMetrics accuracy adapter.

    Metadata:
        - Module type: metric
        - Registry name: ``Accuracy``
        - Task: CLASSIFICATION, SEGMENTATION, ANOMALY_DETECTION
        - Attached node types: None
        - Inputs: ``predictions``, ``target``
        - Outputs: scalar accuracy tensor, or scalar plus per-class sub-metrics
        - State: wrapped ``torchmetrics.Accuracy`` state

    Prediction format:
        ``predictions`` contains logits or scores accepted by
        ``torchmetrics.Accuracy``.

    Target format:
        ``target`` contains labels for classification or segmentation. For
        anomaly detection, segmentation labels are used.

    Formula:
        Delegates accuracy accumulation and computation to
        ``torchmetrics.Accuracy``.

    Provenance:
        - Source: torchmetrics
        - License: Apache License 2.0
        - Implementation notes: Uses ``TorchMetricWrapper`` task inference and
          optional per-class output handling.

    """

    supported_tasks = [
        Tasks.CLASSIFICATION,
        Tasks.SEGMENTATION,
        Tasks.ANOMALY_DETECTION,
    ]
    Metric = torchmetrics.Accuracy


class F1Score(TorchMetricWrapper):
    """TorchMetrics F1 score adapter.

    Metadata:
        - Module type: metric
        - Registry name: ``F1Score``
        - Task: CLASSIFICATION, SEGMENTATION, ANOMALY_DETECTION
        - Attached node types: None
        - Inputs: ``predictions``, ``target``
        - Outputs: scalar F1 tensor, or scalar plus per-class sub-metrics
        - State: wrapped ``torchmetrics.F1Score`` state

    Prediction format:
        ``predictions`` contains logits or scores accepted by
        ``torchmetrics.F1Score``.

    Target format:
        ``target`` contains labels for classification or segmentation. For
        anomaly detection, segmentation labels are used.

    Formula:
        Delegates F1 accumulation and computation to ``torchmetrics.F1Score``.

    Provenance:
        - Source: torchmetrics
        - License: Apache License 2.0
        - Implementation notes: Uses ``TorchMetricWrapper`` task inference and
          optional per-class output handling.

    """

    supported_tasks = [
        Tasks.CLASSIFICATION,
        Tasks.SEGMENTATION,
        Tasks.ANOMALY_DETECTION,
    ]
    Metric = torchmetrics.F1Score


class JaccardIndex(TorchMetricWrapper):
    """TorchMetrics Jaccard index adapter.

    Metadata:
        - Module type: metric
        - Registry name: ``JaccardIndex``
        - Task: CLASSIFICATION, SEGMENTATION, ANOMALY_DETECTION
        - Attached node types: None
        - Inputs: ``predictions``, ``target``
        - Outputs: scalar Jaccard index tensor, or scalar plus per-class
          sub-metrics
        - State: wrapped ``torchmetrics.JaccardIndex`` state

    Prediction format:
        ``predictions`` contains logits or scores accepted by
        ``torchmetrics.JaccardIndex``.

    Target format:
        ``target`` contains labels for classification or segmentation. For
        anomaly detection, segmentation labels are used.

    Formula:
        Delegates Jaccard accumulation and computation to
        ``torchmetrics.JaccardIndex``.

    Provenance:
        - Source: torchmetrics
        - License: Apache License 2.0
        - Implementation notes: Uses ``TorchMetricWrapper`` task inference and
          optional per-class output handling.

    """

    supported_tasks = [
        Tasks.CLASSIFICATION,
        Tasks.SEGMENTATION,
        Tasks.ANOMALY_DETECTION,
    ]
    Metric = torchmetrics.JaccardIndex


class Precision(TorchMetricWrapper):
    """TorchMetrics precision adapter.

    Metadata:
        - Module type: metric
        - Registry name: ``Precision``
        - Task: CLASSIFICATION, SEGMENTATION, ANOMALY_DETECTION
        - Attached node types: None
        - Inputs: ``predictions``, ``target``
        - Outputs: scalar precision tensor, or scalar plus per-class sub-metrics
        - State: wrapped ``torchmetrics.Precision`` state

    Prediction format:
        ``predictions`` contains logits or scores accepted by
        ``torchmetrics.Precision``.

    Target format:
        ``target`` contains labels for classification or segmentation. For
        anomaly detection, segmentation labels are used.

    Formula:
        Delegates precision accumulation and computation to
        ``torchmetrics.Precision``.

    Provenance:
        - Source: torchmetrics
        - License: Apache License 2.0
        - Implementation notes: Uses ``TorchMetricWrapper`` task inference and
          optional per-class output handling.

    """

    supported_tasks = [
        Tasks.CLASSIFICATION,
        Tasks.SEGMENTATION,
        Tasks.ANOMALY_DETECTION,
    ]
    Metric = torchmetrics.Precision


class Recall(TorchMetricWrapper):
    """TorchMetrics recall adapter.

    Metadata:
        - Module type: metric
        - Registry name: ``Recall``
        - Task: CLASSIFICATION, SEGMENTATION, ANOMALY_DETECTION
        - Attached node types: None
        - Inputs: ``predictions``, ``target``
        - Outputs: scalar recall tensor, or scalar plus per-class sub-metrics
        - State: wrapped ``torchmetrics.Recall`` state

    Prediction format:
        ``predictions`` contains logits or scores accepted by
        ``torchmetrics.Recall``.

    Target format:
        ``target`` contains labels for classification or segmentation. For
        anomaly detection, segmentation labels are used.

    Formula:
        Delegates recall accumulation and computation to ``torchmetrics.Recall``.

    Provenance:
        - Source: torchmetrics
        - License: Apache License 2.0
        - Implementation notes: Uses ``TorchMetricWrapper`` task inference and
          optional per-class output handling.

    """

    supported_tasks = [
        Tasks.CLASSIFICATION,
        Tasks.SEGMENTATION,
        Tasks.ANOMALY_DETECTION,
    ]
    Metric = torchmetrics.Recall
