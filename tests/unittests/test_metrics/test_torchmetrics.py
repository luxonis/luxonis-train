import pytest
import torch
import torchmetrics
from torch import Tensor

from luxonis_train.attached_modules.metrics.torchmetrics import (
    TorchMetricWrapper,
)
from luxonis_train.nodes import BaseNode
from luxonis_train.tasks import Tasks
from luxonis_train.utils.dataset_metadata import DatasetMetadata


def test_torchmetrics():
    class DummyNode(BaseNode, register=False):
        task = Tasks.CLASSIFICATION

        def forward(self, _: Tensor) -> Tensor: ...

    class DummyMetric(TorchMetricWrapper):
        supported_tasks = [Tasks.CLASSIFICATION, Tasks.SEGMENTATION]
        Metric = torchmetrics.Accuracy

    node_1_class = DummyNode(n_classes=1)
    node_2_classes = DummyNode(n_classes=2)
    node = DummyNode()
    assert DummyMetric(node=node_1_class)._torchmetric_task == "binary"
    assert DummyMetric(node=node_2_classes)._torchmetric_task == "multiclass"
    assert DummyMetric(node=node_2_classes, task="multilabel")
    assert DummyMetric(num_classes=1)._torchmetric_task == "binary"
    assert DummyMetric(num_classes=2)._torchmetric_task == "multiclass"
    assert DummyMetric(num_labels=2)._torchmetric_task == "multilabel"

    assert DummyMetric(task="binary")

    with pytest.raises(ValueError, match="not possible to infer"):
        DummyMetric()

    with pytest.raises(ValueError, match="Invalid task type 'invalid'"):
        DummyMetric(task="invalid")

    with pytest.raises(ValueError, match="not have the 'num_classes'"):
        DummyMetric(task="multiclass")

    with pytest.raises(ValueError, match="more than 1 class"):
        DummyMetric(task="binary", node=node_2_classes)

    with pytest.raises(ValueError, match="only 1 class"):
        DummyMetric(task="multiclass", node=node_1_class)

    with pytest.raises(ValueError, match="not have the 'num_classes'"):
        DummyMetric(task="multiclass", node=node)

    with pytest.raises(ValueError, match="not have the 'num_labels'"):
        DummyMetric(task="multilabel", node=node)


def test_per_class_torchmetrics():
    class DummyNode(BaseNode, register=False):
        task = Tasks.SEGMENTATION

        def forward(self, _: Tensor) -> Tensor: ...

    class DummyMetricJaccardIndex(TorchMetricWrapper):
        supported_tasks = [Tasks.CLASSIFICATION, Tasks.SEGMENTATION]
        Metric = torchmetrics.JaccardIndex

    predictions = torch.zeros(2, 2, 10, 10, dtype=torch.float)
    targets = torch.zeros(2, 2, 10, 10, dtype=torch.float)

    predictions[0, 0, 1:4, 1:4] = 1.0
    targets[0, 0, 2:5, 2:5] = 1.0

    predictions[1, 1, 5:8, 5:8] = 1.0
    targets[1, 1, 6:9, 6:9] = 1.0

    metric = DummyMetricJaccardIndex(
        task="multiclass",
        node=DummyNode(
            n_classes=2,
            dataset_metadata=DatasetMetadata(
                classes={"": {"class1": 0, "class2": 1}}
            ),
        ),
        average=None,
    )
    metric.update(predictions, targets)
    result = metric.compute()

    assert isinstance(result, tuple)
    assert len(result) == 2
    assert result[1]["MulticlassJaccardIndex_class1"].ndim == 0
    assert result[1]["MulticlassJaccardIndex_class2"].ndim == 0
