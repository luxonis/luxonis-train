import pytest
import torch
from torch import Tensor

from luxonis_train import BaseLoss, BaseNode
from luxonis_train.enums import Task
from luxonis_train.utils.exceptions import IncompatibleException
from luxonis_train.utils.types import Labels, Packet

SEGMENTATION_ARRAY = torch.tensor([0])
KEYPOINT_ARRAY = torch.tensor([1])
BOUNDINGBOX_ARRAY = torch.tensor([2])
CLASSIFICATION_ARRAY = torch.tensor([3])
FEATURES_ARRAY = torch.tensor([4])


class DummyBackbone(BaseNode):
    def forward(self, _): ...


class DummySegmentationHead(BaseNode):
    tasks = [Task.SEGMENTATION]

    def forward(self, _): ...


class DummyBBoxHead(BaseNode):
    tasks = [Task.BOUNDINGBOX]

    def forward(self, _): ...


class DummyDetectionHead(BaseNode):
    tasks = [Task.BOUNDINGBOX, Task.KEYPOINTS]

    def forward(self, _): ...


class DummyLoss(BaseLoss):
    supported_tasks = [
        Task.SEGMENTATION,
        (Task.KEYPOINTS, Task.BOUNDINGBOX),
    ]

    def forward(self, _): ...


class NoLabelLoss(BaseLoss):
    def forward(self, _): ...


@pytest.fixture
def labels() -> Labels:
    return {
        "/segmentation": SEGMENTATION_ARRAY,
        "/keypoints": KEYPOINT_ARRAY,
        "/boundingbox": BOUNDINGBOX_ARRAY,
        "/classification": CLASSIFICATION_ARRAY,
    }


@pytest.fixture
def inputs() -> Packet[Tensor]:
    return {
        "features": [FEATURES_ARRAY],
        "/segmentation": [SEGMENTATION_ARRAY],
    }


def test_valid_properties():
    head = DummySegmentationHead()
    loss = DummyLoss(node=head)
    no_labels_loss = NoLabelLoss(node=head)
    assert loss.node == head
    assert loss.node_tasks == [Task.SEGMENTATION]
    assert loss.required_labels == [Task.SEGMENTATION]
    assert no_labels_loss.node == head
    assert no_labels_loss.node_tasks == [Task.SEGMENTATION]
    assert no_labels_loss.required_labels == []


def test_invalid_properties():
    backbone = DummyBackbone()
    with pytest.raises(IncompatibleException):
        DummyLoss(node=backbone)
    with pytest.raises(IncompatibleException):
        DummyLoss(node=DummyBBoxHead())
    with pytest.raises(RuntimeError):
        _ = DummyLoss().node
    with pytest.raises(RuntimeError):
        _ = NoLabelLoss(node=backbone).node_tasks


def test_get_label(labels: Labels):
    seg_head = DummySegmentationHead()
    det_head = DummyDetectionHead()
    seg_loss = DummyLoss(node=seg_head)
    assert seg_loss.get_label(labels) == SEGMENTATION_ARRAY
    assert seg_loss.get_label(labels, Task.SEGMENTATION) == SEGMENTATION_ARRAY

    del labels["/segmentation"]
    labels["task/segmentation"] = SEGMENTATION_ARRAY

    with pytest.raises(IncompatibleException):
        seg_loss.get_label(labels)

    det_loss = DummyLoss(node=det_head)
    assert det_loss.get_label(labels, Task.KEYPOINTS) == KEYPOINT_ARRAY
    assert det_loss.get_label(labels, Task.BOUNDINGBOX) == BOUNDINGBOX_ARRAY

    with pytest.raises(ValueError):
        det_loss.get_label(labels)

    with pytest.raises(IncompatibleException):
        det_loss.get_label(labels, Task.SEGMENTATION)


def test_input_tensors(inputs: Packet[Tensor]):
    seg_head = DummySegmentationHead()
    seg_loss = DummyLoss(node=seg_head)
    assert seg_loss.pick_inputs(inputs) == [SEGMENTATION_ARRAY]
    assert seg_loss.pick_inputs(inputs, "/segmentation") == [
        SEGMENTATION_ARRAY
    ]
    assert seg_loss.pick_inputs(inputs, Task.SEGMENTATION) == [
        SEGMENTATION_ARRAY
    ]

    with pytest.raises(IncompatibleException):
        seg_loss.pick_inputs(inputs, Task.KEYPOINTS)
    with pytest.raises(IncompatibleException):
        seg_loss.pick_inputs(inputs, "/keypoints")

    det_head = DummyDetectionHead()
    det_loss = DummyLoss(node=det_head)
    with pytest.raises(ValueError):
        det_loss.pick_inputs(inputs)


def test_prepare(inputs: Packet[Tensor], labels: Labels):
    backbone = DummyBackbone()
    seg_head = DummySegmentationHead()
    seg_loss = DummyLoss(node=seg_head)
    det_head = DummyDetectionHead()

    assert seg_loss.prepare(inputs, labels) == (
        [SEGMENTATION_ARRAY],
        SEGMENTATION_ARRAY,
    )
    inputs["/segmentation"].append(FEATURES_ARRAY)
    assert seg_loss.prepare(inputs, labels) == (
        [SEGMENTATION_ARRAY, FEATURES_ARRAY],
        SEGMENTATION_ARRAY,
    )

    with pytest.raises(RuntimeError):
        NoLabelLoss(node=backbone).prepare(inputs, labels)

    with pytest.raises(RuntimeError):
        NoLabelLoss(node=seg_head).prepare(inputs, labels)

    with pytest.raises(RuntimeError):
        DummyLoss(node=det_head).prepare(inputs, labels)
