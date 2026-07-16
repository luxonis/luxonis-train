from typing import Any, cast

from torch import Tensor, nn

from luxonis_train import BaseNode
from luxonis_train.attached_modules.metrics import MeanAveragePrecision, MIoU
from luxonis_train.config import NodeConfig
from luxonis_train.lightning.utils import (
    NodeWrapper,
    _translate_predefined_metric_params,
)
from luxonis_train.tasks import Task, Tasks


class DummyNode(BaseNode):
    def __init__(self, task: Task):
        super().__init__()
        self.task = task

    def forward(self, _: Tensor) -> Tensor: ...


def test_translate_predefined_metric_params_detection_map():
    params = _translate_predefined_metric_params(
        DummyNode(Tasks.BOUNDINGBOX),
        "MeanAveragePrecision",
        MeanAveragePrecision,
        {"per_class_metrics": True},
    )

    assert params == {"class_metrics": True}


def test_translate_predefined_metric_params_keypoint_map():
    params = _translate_predefined_metric_params(
        DummyNode(Tasks.INSTANCE_KEYPOINTS),
        "MeanAveragePrecision",
        MeanAveragePrecision,
        {"per_class_metrics": True},
    )

    assert params == {"class_metrics": True}


def test_translate_predefined_metric_params_segmentation_iou():
    params = _translate_predefined_metric_params(
        DummyNode(Tasks.SEGMENTATION),
        "MIoU",
        MIoU,
        {"num_classes": 3, "per_class_metrics": True},
    )

    assert params == {"num_classes": 3, "per_class": True}


def test_translate_predefined_metric_params_segmentation_iou_false():
    params = _translate_predefined_metric_params(
        DummyNode(Tasks.SEGMENTATION),
        "MIoU",
        MIoU,
        {"num_classes": 3, "per_class_metrics": False},
    )

    assert params == {"num_classes": 3, "per_class": False}


def test_node_wrapper_train_updates_self_and_attached_modules():
    node = DummyNode(Tasks.CLASSIFICATION)
    loss = nn.Dropout()
    metric = nn.Dropout()
    visualizer = nn.Dropout()
    wrapper = NodeWrapper(
        name="node",
        module=node,
        losses=cast(Any, {"loss": loss}),
        metrics=cast(Any, {"metric": metric}),
        visualizers=cast(Any, {"visualizer": visualizer}),
        unfreeze_after=None,
        lr_after_unfreeze=None,
        cfg=NodeConfig(name="DummyNode"),
    )

    wrapper.eval()
    assert wrapper.training is False
    assert node.training is False
    assert loss.training is False
    assert metric.training is False
    assert visualizer.training is False

    wrapper.train()
    assert wrapper.training is True
    assert node.training is True
    assert loss.training is True
    assert metric.training is True
    assert visualizer.training is True
