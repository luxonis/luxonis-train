from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from torch import Tensor, nn

from luxonis_train import BaseNode
from luxonis_train.attached_modules.metrics import MeanAveragePrecision, MIoU
from luxonis_train.lightning.luxonis_lightning import _prepare_balanced_labels
from luxonis_train.lightning.utils import (
    NodeWrapper,
    _translate_predefined_metric_params,
    get_model_execution_order,
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
        finetuning=[],
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


def test_prepare_balanced_labels_skips_unpaired_segmentation_tasks():
    labels = {
        "paired/segmentation": torch.zeros(1, 2, 2, 2),
        "paired/classification": torch.tensor([[0, 1, 0]]),
        "unpaired/segmentation": torch.zeros(1, 2, 2, 2),
        "other/classification": torch.tensor([[0, 1]]),
    }

    prepared = _prepare_balanced_labels(labels)

    assert torch.equal(
        prepared["paired/classification"],
        labels["paired/classification"][:, 1:],
    )
    assert torch.equal(
        prepared["other/classification"], labels["other/classification"]
    )


@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("fail_forward", [False, True])
def test_execution_order_preserves_modes_and_cleans_up_hooks(
    training: bool,
    fail_forward: bool,
):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = NodeWrapper(
                name="layers",
                module=cast(
                    Any, nn.Sequential(nn.Conv2d(3, 4, 1), nn.BatchNorm2d(4))
                ),
                losses=cast(Any, {"loss": nn.Dropout()}),
                metrics=cast(Any, {"metric": nn.Dropout()}),
                visualizers={},
                unfreeze_after=None,
                lr_after_unfreeze=None,
                finetuning=[],
            )
            self.nodes = SimpleNamespace(
                loader_input_shapes={"layers": {"image": (3, 4, 4)}}
            )
            self.device = torch.device("cpu")

        def forward(self, inputs: dict[str, Tensor]) -> Tensor:
            result = self.layers.module(inputs["image"])
            if fail_forward:
                raise RuntimeError("forward failed")
            return result

    model = Model().train(training)
    model.layers.module[1].eval()
    model.layers.metrics["metric"].eval()
    loss_training = model.layers.losses["loss"].training
    modes = {name: layer.training for name, layer in model.named_modules()}
    if fail_forward:
        with pytest.raises(RuntimeError, match="forward failed"):
            get_model_execution_order(cast(Any, model))
    else:
        assert get_model_execution_order(cast(Any, model)) == [
            "layers.module.0",
            "layers.module.1",
        ]
    assert {
        name: layer.training for name, layer in model.named_modules()
    } == modes
    assert all(not layer._forward_hooks for layer in model.modules())
    assert model.layers.losses["loss"].training == loss_training
    assert model.layers.metrics["metric"].training is False
