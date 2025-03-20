import pytest
import torchmetrics
from torch import Tensor

from luxonis_train.attached_modules.metrics.torchmetrics import (
    TorchMetricWrapper,
)
from luxonis_train.nodes import BaseNode
from luxonis_train.tasks import Tasks


def test_torchmetrics():
    class DummyNode(BaseNode):
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
