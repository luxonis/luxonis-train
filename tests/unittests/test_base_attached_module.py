import pytest
from torch import Tensor

from luxonis_train import BaseHead, BaseLoss, BaseNode
from luxonis_train.tasks import Tasks
from luxonis_train.utils.exceptions import IncompatibleError


class DummyBackbone(BaseNode):
    def forward(self, _: Tensor) -> Tensor: ...


class DummySegmentationHead(BaseHead):
    task = Tasks.SEGMENTATION

    def forward(self, _: Tensor) -> Tensor: ...


class DummyBBoxHead(BaseHead):
    task = Tasks.BOUNDINGBOX

    def forward(self, _: Tensor) -> Tensor: ...


class DummyDetectionHead(BaseHead):
    task = Tasks.INSTANCE_KEYPOINTS

    def forward(self, _: Tensor) -> Tensor: ...


class DummyLoss(BaseLoss):
    supported_tasks = [Tasks.SEGMENTATION, Tasks.INSTANCE_KEYPOINTS]

    def forward(self, _: Tensor) -> Tensor: ...


class NoLabelLoss(BaseLoss):
    def forward(self, _: Tensor) -> Tensor: ...


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
    with pytest.raises(IncompatibleError):
        DummyLoss(node=DummyBBoxHead())
    with pytest.raises(RuntimeError):
        _ = DummyLoss().node
    with pytest.raises(RuntimeError):
        _ = NoLabelLoss(node=backbone).task
