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
    compute_visualization_buffer,
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
                visualizers=cast(
                    Any,
                    {"visualizer": nn.Sequential(nn.Dropout(), nn.Dropout())},
                ),
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
    batchnorm = cast(
        nn.BatchNorm2d, cast(nn.Sequential, model.layers.module)[1]
    )
    batchnorm.eval()
    model.layers.metrics["metric"].eval()
    cast(nn.Sequential, model.layers.visualizers["visualizer"])[1].eval()
    loss_training = model.layers.losses["loss"].training
    visualizer_modes = [
        module.training
        for module in model.layers.visualizers["visualizer"].modules()
    ]
    running_mean = cast(Tensor, batchnorm.running_mean).clone()
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
    assert [
        module.training
        for module in model.layers.visualizers["visualizer"].modules()
    ] == visualizer_modes
    torch.testing.assert_close(batchnorm.running_mean, running_mean)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_visualization_buffer_is_bounded_and_detached(device: str):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    images = torch.arange(48.0, device=device).reshape(4, 3, 2, 2)
    images.requires_grad_()
    batch = {"head": {"boxes": images, "keypoints": images + 1}}
    buffer = []
    for _ in range(20):
        extra = compute_visualization_buffer(buffer, batch, [0, 2], 5)
        if extra is not None:
            buffer.append(extra)
    assert [entry["head"]["boxes"].shape[0] for entry in buffer] == [2, 2, 1]
    for entry in buffer:
        for name, tensor in entry["head"].items():
            assert tensor.device.type == "cpu"
            assert not tensor.requires_grad
            assert tensor.grad_fn is None
            expected = batch["head"][name][[1, 3]][: tensor.shape[0]]
            torch.testing.assert_close(tensor, expected.detach().cpu())


@pytest.mark.parametrize(("max_log_images", "logged"), [(0, []), (3, [0, 1])])
def test_visualization_buffer_skips_disabled_or_fully_logged_batches(
    max_log_images: int, logged: list[int]
):
    assert (
        compute_visualization_buffer(
            [],
            {"head": {"boxes": torch.zeros(2, 3, 2, 2)}},
            logged,
            max_log_images,
        )
        is None
    )
