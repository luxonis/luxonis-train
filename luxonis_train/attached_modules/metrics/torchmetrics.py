"""Metrics that wrap the classification metrics of ``torchmetrics``.

`TorchMetricWrapper` infers ``task`` and the number of classes from the
node that the metric attaches to, when the config does not set them. The
other arguments go to the ``torchmetrics`` metric, such as ``average``.

"""

from contextlib import suppress
from functools import cached_property

import torchmetrics
from torch import Tensor
from typing_extensions import override

from luxonis_train.tasks import Metadata, Tasks

from .base_metric import BaseMetric


class TorchMetricWrapper(BaseMetric):
    """Base class for the metrics that wrap a ``torchmetrics`` metric.

    A subclass sets the class attribute ``Metric`` to a task wrapper of
    ``torchmetrics``, such as ``torchmetrics.Accuracy``. `__init__`
    resolves ``task`` and the number of classes, and stores the built
    metric in the ``metric`` attribute. `update`, `compute`, and `reset`
    call the built metric. This class does not set ``Metric``, so a
    config that names it fails with ``AttributeError``. Name a subclass
    instead.

    Inputs:
        - ``predictions`` (``Tensor``): ``[B, n_classes, ...]`` logits
          or probabilities
        - ``target`` (``Tensor``): ``[B, n_classes, ...]`` one-hot or
          multi-hot labels

    Outputs:
        - ``Tensor``: scalar
        - ``tuple[Tensor, dict[str, Tensor]]``: the mean and the value
          of each class, when the built metric returns one value for
          each class, for example with ``average: "none"``

    References:
        - Source: Wraps `torchmetrics
          <https://github.com/Lightning-AI/torchmetrics>`_ (Apache-2.0).
        - License: Apache-2.0 (this project)

    Notes:
        ``task`` is ``"binary"``, ``"multiclass"``, or ``"multilabel"``.
        When the ``params`` do not set it, the metric infers it and logs
        a warning. For ``"multiclass"``, `update` converts the one-hot
        target to class indices. On an ``anomaly_detection`` node, the
        metric reads the ``segmentation`` label.

    Example:
        A subclass, here `Accuracy`, attached to a ``DiscSubNetHead`` in
        ``model.nodes``:

        .. code-block:: yaml

            - name: DiscSubNetHead
              inputs: [RecSubNet]
              metrics:
                - name: Accuracy

    Compatible with:
        - Nodes:

          - `BiSeNetHead`
          - `DDRNetSegmentationHead`
          - `DiscSubNetHead`
          - `SegmentationHead`
          - `TransformerSegmentationHead`

    """

    Metric: type[torchmetrics.Metric]

    def __init__(self, **kwargs):
        """Resolve the task and build the wrapped ``torchmetrics`` metric.

        ``task`` comes from ``kwargs``. Without it, the method infers
        ``"binary"`` from ``num_classes`` of ``1``, ``"multiclass"``
        from another ``num_classes``, and ``"multilabel"`` from
        ``num_labels``. Without these arguments, it infers ``"binary"``
        for a node with one class and ``"multiclass"`` for other nodes.
        It logs a warning when it infers the task.

        The number of classes comes from ``num_classes``, then from
        ``num_labels``, then from the node. The method passes it to
        ``Metric`` as ``num_classes`` for ``"multiclass"``, and as
        ``num_labels`` for ``"multilabel"``.

        Args:
            **kwargs (``Any``): ``node`` goes to `BaseMetric`. The other
                arguments and the resolved ``task`` go to the constructor
                of ``Metric``, for example ``average`` or ``threshold``.

        Raises:
            ValueError: When the method cannot infer ``task``, or when
                ``task`` is not ``"binary"``, ``"multiclass"``, or
                ``"multilabel"``. Also when the number of classes does
                not fit the task: unknown or ``1`` for ``"multiclass"``
                and ``"multilabel"``, or more than ``1`` for ``"binary"``.

        Example:
            >>> metric = Accuracy(task="multilabel", num_labels=3)
            >>> type(metric.metric).__name__
            'MultilabelAccuracy'

        """
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
        """Add one batch to the wrapped metric.

        For ``"multiclass"``, the method converts ``target`` to class
        indices with ``argmax`` over dimension ``1``. For the other
        tasks, it passes ``target`` unchanged.

        Args:
            predictions (``Tensor``): The main output of the node, of
                shape ``[B, n_classes, ...]``. Logits or probabilities.
            target (``Tensor``): The label of the task, of shape
                ``[B, n_classes, ...]``. It is a binary mask for
                ``"binary"``, one-hot for ``"multiclass"``, and multi-hot
                for ``"multilabel"``.

        """
        if self._torchmetric_task == "multiclass":
            target = target.argmax(dim=1)
        self.metric.update(predictions, target)

    @override
    def compute(self) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        """Compute the value of the wrapped metric.

        The method returns a result with one element unchanged. It
        treats a result with more than one element as one value for each
        class, as with ``average: "none"``. It then returns the mean of
        the values and a dictionary of the class values. The keys are
        ``"<Metric>_<class name>"``, where ``<Metric>`` is the class name
        of the built metric, such as ``MulticlassAccuracy``. The class
        names come from the node. Without a node, the ``classes``
        property raises ``RuntimeError``.

        Returns:
            ``Tensor | tuple[Tensor, dict[str, Tensor]]``: The scalar
            value, or the mean of the class values and the dictionary of
            the class values.

        Raises:
            ValueError: When the result has more than one element and
                the node has no class names.

        Example:
            One correct prediction out of two:

            >>> import torch
            >>> metric = Accuracy(task="multiclass", num_classes=3)
            >>> predictions = torch.tensor([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
            >>> target = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
            >>> metric.update(predictions, target)
            >>> metric.compute().item()
            0.5

        """
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
        """Reset the states of the wrapped metric.

        The method calls ``reset`` only on the ``metric`` attribute. It
        does not call the ``reset`` of the ``torchmetrics`` base class
        on the wrapper. Thus the wrapper keeps the cached result of its
        last `compute`, and `compute` returns that result until the next
        `update`.

        """
        self.metric.reset()

    @cached_property
    @override
    def required_labels(self) -> set[str | Metadata]:
        """The labels for the ``target`` parameter of `update`.

        `BaseAttachedModule.get_parameters` reads this set for the
        ``target`` parameter of `update`, which has no label suffix. On
        an ``anomaly_detection`` node, the set holds only
        ``"segmentation"``, so ``target`` receives the anomaly mask. On
        other nodes, it holds the labels of the task. The property
        raises ``RuntimeError`` when the metric has no task.

        """
        if self.task == Tasks.ANOMALY_DETECTION:
            return Tasks.SEGMENTATION.required_labels
        return self.task.required_labels


class Accuracy(TorchMetricWrapper):
    r"""Accuracy metric that wraps ``torchmetrics.Accuracy``.

    Inputs:
        - ``predictions`` (``Tensor``): ``[B, n_classes, ...]`` logits
          or probabilities
        - ``target`` (``Tensor``): ``[B, n_classes, ...]`` one-hot or
          multi-hot labels

    Outputs:
        - ``Tensor``: scalar
        - ``tuple[Tensor, dict[str, Tensor]]``: the mean and the value
          of each class, when the built metric returns one value for
          each class, for example with ``average: "none"``

    Formula:
        For ``"binary"``, with the counts of true and false positives
        and negatives:

        .. math::

            \text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}

        For ``"multiclass"`` and ``"multilabel"``,
        ``torchmetrics.Accuracy`` counts the statistics of each class and
        combines the classes as its ``average`` argument selects.

    References:
        - Source: Wraps `torchmetrics
          <https://github.com/Lightning-AI/torchmetrics>`_ (Apache-2.0).
        - License: Apache-2.0 (this project)

    Notes:
        `TorchMetricWrapper` resolves ``task`` and the number of
        classes. The other ``params`` go to ``torchmetrics.Accuracy``,
        for example ``average``, ``threshold``, or ``top_k``.

    Example:
        Attached to a ``ClassificationHead`` in ``model.nodes``:

        .. code-block:: yaml

            - name: ClassificationHead
              inputs: [ResNet]
              metrics:
                - name: Accuracy

    Compatible with:
        - Used by: `ClassificationModel`
        - Nodes:

          - `BiSeNetHead`
          - `ClassificationHead`
          - `DDRNetSegmentationHead`
          - `DiscSubNetHead`
          - `SegmentationHead`
          - `TransformerClassificationHead`
          - `TransformerSegmentationHead`

    """

    supported_tasks = [
        Tasks.CLASSIFICATION,
        Tasks.SEGMENTATION,
        Tasks.ANOMALY_DETECTION,
    ]
    Metric = torchmetrics.Accuracy


class F1Score(TorchMetricWrapper):
    r"""F1 score metric that wraps ``torchmetrics.F1Score``.

    Inputs:
        - ``predictions`` (``Tensor``): ``[B, n_classes, ...]`` logits
          or probabilities
        - ``target`` (``Tensor``): ``[B, n_classes, ...]`` one-hot or
          multi-hot labels

    Outputs:
        - ``Tensor``: scalar
        - ``tuple[Tensor, dict[str, Tensor]]``: the mean and the value
          of each class, when the built metric returns one value for
          each class, for example with ``average: "none"``

    Formula:
        With the counts of true positives, false positives, and false
        negatives:

        .. math::

            F_1 = \frac{2\,TP}{2\,TP + FP + FN}

        For ``"multiclass"`` and ``"multilabel"``,
        ``torchmetrics.F1Score`` counts the statistics of each class and
        combines the classes as its ``average`` argument selects.

    References:
        - Source: Wraps `torchmetrics
          <https://github.com/Lightning-AI/torchmetrics>`_ (Apache-2.0).
        - License: Apache-2.0 (this project)

    Notes:
        `TorchMetricWrapper` resolves ``task`` and the number of
        classes. The other ``params`` go to ``torchmetrics.F1Score``,
        for example ``average``, ``threshold``, or ``top_k``.

    Example:
        Attached to a ``ClassificationHead`` in ``model.nodes``:

        .. code-block:: yaml

            - name: ClassificationHead
              inputs: [ResNet]
              metrics:
                - name: F1Score

    Compatible with:
        - Used by:

          - `ClassificationModel`
          - `SegmentationModel`

        - Nodes:

          - `BiSeNetHead`
          - `ClassificationHead`
          - `DDRNetSegmentationHead`
          - `DiscSubNetHead`
          - `SegmentationHead`
          - `TransformerClassificationHead`
          - `TransformerSegmentationHead`

    """

    supported_tasks = [
        Tasks.CLASSIFICATION,
        Tasks.SEGMENTATION,
        Tasks.ANOMALY_DETECTION,
    ]
    Metric = torchmetrics.F1Score


class JaccardIndex(TorchMetricWrapper):
    r"""Jaccard index metric that wraps ``torchmetrics.JaccardIndex``.

    Inputs:
        - ``predictions`` (``Tensor``): ``[B, n_classes, ...]`` logits
          or probabilities
        - ``target`` (``Tensor``): ``[B, n_classes, ...]`` one-hot or
          multi-hot labels

    Outputs:
        - ``Tensor``: scalar
        - ``tuple[Tensor, dict[str, Tensor]]``: the mean and the value
          of each class, when the built metric returns one value for
          each class, for example with ``average: "none"``

    Formula:
        With the counts of true positives, false positives, and false
        negatives:

        .. math::

            J = \frac{TP}{TP + FP + FN}

        For ``"multiclass"`` and ``"multilabel"``,
        ``torchmetrics.JaccardIndex`` counts the statistics of each
        class and combines the classes as its ``average`` argument
        selects.

    References:
        - Source: Wraps `torchmetrics
          <https://github.com/Lightning-AI/torchmetrics>`_ (Apache-2.0).
        - License: Apache-2.0 (this project)

    Notes:
        `TorchMetricWrapper` resolves ``task`` and the number of
        classes. The other ``params`` go to
        ``torchmetrics.JaccardIndex``, for example ``average``,
        ``threshold``, or ``ignore_index``.

    Example:
        Attached to a ``ClassificationHead`` in ``model.nodes``:

        .. code-block:: yaml

            - name: ClassificationHead
              inputs: [ResNet]
              metrics:
                - name: JaccardIndex

    Compatible with:
        - Used by:

          - `AnomalyDetectionModel`
          - `SegmentationModel`

        - Nodes:

          - `BiSeNetHead`
          - `ClassificationHead`
          - `DDRNetSegmentationHead`
          - `DiscSubNetHead`
          - `SegmentationHead`
          - `TransformerClassificationHead`
          - `TransformerSegmentationHead`

    """

    supported_tasks = [
        Tasks.CLASSIFICATION,
        Tasks.SEGMENTATION,
        Tasks.ANOMALY_DETECTION,
    ]
    Metric = torchmetrics.JaccardIndex


class Precision(TorchMetricWrapper):
    r"""Precision metric that wraps ``torchmetrics.Precision``.

    Inputs:
        - ``predictions`` (``Tensor``): ``[B, n_classes, ...]`` logits
          or probabilities
        - ``target`` (``Tensor``): ``[B, n_classes, ...]`` one-hot or
          multi-hot labels

    Outputs:
        - ``Tensor``: scalar
        - ``tuple[Tensor, dict[str, Tensor]]``: the mean and the value
          of each class, when the built metric returns one value for
          each class, for example with ``average: "none"``

    Formula:
        With the counts of true positives and false positives:

        .. math::

            \text{Precision} = \frac{TP}{TP + FP}

        For ``"multiclass"`` and ``"multilabel"``,
        ``torchmetrics.Precision`` counts the statistics of each class
        and combines the classes as its ``average`` argument selects.

    References:
        - Source: Wraps `torchmetrics
          <https://github.com/Lightning-AI/torchmetrics>`_ (Apache-2.0).
        - License: Apache-2.0 (this project)

    Notes:
        `TorchMetricWrapper` resolves ``task`` and the number of
        classes. The other ``params`` go to ``torchmetrics.Precision``,
        for example ``average``, ``threshold``, or ``top_k``.

    Example:
        Attached to a ``ClassificationHead`` in ``model.nodes``:

        .. code-block:: yaml

            - name: ClassificationHead
              inputs: [ResNet]
              metrics:
                - name: Precision

    Compatible with:
        - Nodes:

          - `BiSeNetHead`
          - `ClassificationHead`
          - `DDRNetSegmentationHead`
          - `DiscSubNetHead`
          - `SegmentationHead`
          - `TransformerClassificationHead`
          - `TransformerSegmentationHead`

    """

    supported_tasks = [
        Tasks.CLASSIFICATION,
        Tasks.SEGMENTATION,
        Tasks.ANOMALY_DETECTION,
    ]
    Metric = torchmetrics.Precision


class Recall(TorchMetricWrapper):
    r"""Recall metric that wraps ``torchmetrics.Recall``.

    Inputs:
        - ``predictions`` (``Tensor``): ``[B, n_classes, ...]`` logits
          or probabilities
        - ``target`` (``Tensor``): ``[B, n_classes, ...]`` one-hot or
          multi-hot labels

    Outputs:
        - ``Tensor``: scalar
        - ``tuple[Tensor, dict[str, Tensor]]``: the mean and the value
          of each class, when the built metric returns one value for
          each class, for example with ``average: "none"``

    Formula:
        With the counts of true positives and false negatives:

        .. math::

            \text{Recall} = \frac{TP}{TP + FN}

        For ``"multiclass"`` and ``"multilabel"``,
        ``torchmetrics.Recall`` counts the statistics of each class and
        combines the classes as its ``average`` argument selects.

    References:
        - Source: Wraps `torchmetrics
          <https://github.com/Lightning-AI/torchmetrics>`_ (Apache-2.0).
        - License: Apache-2.0 (this project)

    Notes:
        `TorchMetricWrapper` resolves ``task`` and the number of
        classes. The other ``params`` go to ``torchmetrics.Recall``, for
        example ``average``, ``threshold``, or ``top_k``.

    Example:
        Attached to a ``ClassificationHead`` in ``model.nodes``:

        .. code-block:: yaml

            - name: ClassificationHead
              inputs: [ResNet]
              metrics:
                - name: Recall

    Compatible with:
        - Used by: `ClassificationModel`
        - Nodes:

          - `BiSeNetHead`
          - `ClassificationHead`
          - `DDRNetSegmentationHead`
          - `DiscSubNetHead`
          - `SegmentationHead`
          - `TransformerClassificationHead`
          - `TransformerSegmentationHead`

    """

    supported_tasks = [
        Tasks.CLASSIFICATION,
        Tasks.SEGMENTATION,
        Tasks.ANOMALY_DETECTION,
    ]
    Metric = torchmetrics.Recall
