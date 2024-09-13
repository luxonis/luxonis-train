import pytest
import torchmetrics
from luxonis_ml.data import LabelType

from luxonis_train.attached_modules.metrics.torchmetrics import (
    TorchMetricWrapper,
)
from luxonis_train.nodes import BaseNode


def test_torchmetrics():
    class DummyNode(BaseNode):
        tasks = [LabelType.CLASSIFICATION, LabelType.SEGMENTATION]

        def forward(self, _): ...

    class DummyMetric(TorchMetricWrapper):
        supported_labels = [LabelType.CLASSIFICATION, LabelType.SEGMENTATION]
        Metric = torchmetrics.Accuracy

    node_1_class = DummyNode(n_classes=1)
    node_2_classes = DummyNode(n_classes=2)
    assert DummyMetric(node=node_1_class)._task == "binary"
    assert DummyMetric(node=node_2_classes)._task == "multiclass"
    assert DummyMetric(task="binary")

    with pytest.raises(ValueError):
        DummyMetric()

    with pytest.raises(ValueError):
        DummyMetric(task="multiclass")
