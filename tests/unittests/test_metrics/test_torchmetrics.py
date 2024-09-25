import pytest
import torchmetrics

from luxonis_train.attached_modules.metrics.torchmetrics import (
    TorchMetricWrapper,
)
from luxonis_train.enums import TaskType
from luxonis_train.nodes import BaseNode


def test_torchmetrics():
    class DummyNode(BaseNode):
        tasks = [TaskType.CLASSIFICATION, TaskType.SEGMENTATION]

        def forward(self, _): ...

    class DummyMetric(TorchMetricWrapper):
        supported_tasks: list[TaskType] = [
            TaskType.CLASSIFICATION,
            TaskType.SEGMENTATION,
        ]
        Metric = torchmetrics.Accuracy

    node_1_class = DummyNode(n_classes=1)
    node_2_classes = DummyNode(n_classes=2)
    node = DummyNode()
    assert DummyMetric(node=node_1_class)._task == "binary"
    assert DummyMetric(node=node_2_classes)._task == "multiclass"
    assert DummyMetric(node=node_2_classes, task="multilabel")
    assert DummyMetric(num_classes=1)._task == "binary"
    assert DummyMetric(num_classes=2)._task == "multiclass"
    assert DummyMetric(num_labels=2)._task == "multilabel"

    assert DummyMetric(task="binary")

    with pytest.raises(ValueError):
        DummyMetric()

    with pytest.raises(ValueError):
        DummyMetric(task="multiclass")

    with pytest.raises(ValueError):
        DummyMetric(task="invalid")

    with pytest.raises(ValueError):
        DummyMetric(task="binary", node=node_2_classes)

    with pytest.raises(ValueError):
        DummyMetric(task="multiclass", node=node_1_class)

    with pytest.raises(ValueError):
        DummyMetric(task="multiclass", node=node)

    with pytest.raises(ValueError):
        DummyMetric(task="multilabel", node=node)
