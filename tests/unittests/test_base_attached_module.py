import pytest
import torch
from torch import Tensor

from luxonis_train import BaseLoss, BaseNode
from luxonis_train.tasks import Tasks
from luxonis_train.utils import Labels, Packet
from luxonis_train.utils.exceptions import IncompatibleException

SEGMENTATION_ARRAY = torch.tensor([0])
KEYPOINT_ARRAY = torch.tensor([1])
BOUNDINGBOX_ARRAY = torch.tensor([2])
CLASSIFICATION_ARRAY = torch.tensor([3])
FEATURES_ARRAY = torch.tensor([4])


class DummyBackbone(BaseNode):
    def forward(self, _: Tensor) -> Tensor: ...


class DummySegmentationHead(BaseNode):
    task = Tasks.SEGMENTATION

    def forward(self, _: Tensor) -> Tensor: ...


class DummyBBoxHead(BaseNode):
    task = Tasks.BOUNDINGBOX

    def forward(self, _: Tensor) -> Tensor: ...


class DummyDetectionHead(BaseNode):
    task = Tasks.INSTANCE_KEYPOINTS

    def forward(self, _: Tensor) -> Tensor: ...


class DummyLoss(BaseLoss):
    supported_tasks = [Tasks.SEGMENTATION, Tasks.INSTANCE_KEYPOINTS]

    def forward(self, _: Tensor) -> Tensor: ...


class NoLabelLoss(BaseLoss):
    def forward(self, _: Tensor) -> Tensor: ...


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
        "segmentation": [SEGMENTATION_ARRAY],
    }


def test_valid_properties():
    head = DummySegmentationHead()
    loss = DummyLoss(node=head)
    no_labels_loss = NoLabelLoss(node=head)
    assert loss.node is head
    assert loss.task == Tasks.SEGMENTATION
    assert loss.required_labels == {"segmentation"}
    assert no_labels_loss.node is head
    assert no_labels_loss.task == Tasks.SEGMENTATION
    assert no_labels_loss.required_labels == {"segmentation"}


def test_invalid_properties():
    backbone = DummyBackbone()
    with pytest.raises(IncompatibleException):
        DummyLoss(node=DummyBBoxHead())
    with pytest.raises(RuntimeError):
        _ = DummyLoss().node
    with pytest.raises(RuntimeError):
        _ = NoLabelLoss(node=backbone).task


def test_pick_labels(labels: Labels):
    seg_head = DummySegmentationHead()
    det_head = DummyDetectionHead()
    seg_loss = DummyLoss(node=seg_head)
    assert seg_loss.pick_labels(labels) == {"segmentation": SEGMENTATION_ARRAY}

    del labels["/segmentation"]
    labels["task/segmentation"] = SEGMENTATION_ARRAY

    det_loss = DummyLoss(node=det_head)
    assert det_loss.pick_labels(labels) == {
        "keypoints": KEYPOINT_ARRAY,
        "boundingbox": BOUNDINGBOX_ARRAY,
    }


def test_pick_inputs(inputs: Packet[Tensor]):
    seg_head = DummySegmentationHead()
    seg_loss = DummyLoss(node=seg_head)
    assert seg_loss.pick_inputs(inputs, {"segmentation"}) == {
        "segmentation": [SEGMENTATION_ARRAY]
    }

    with pytest.raises(RuntimeError):
        seg_loss.pick_inputs(inputs, {"keypoints"})
