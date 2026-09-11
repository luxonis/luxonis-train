import pytest
import torch
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


def test_wildcard_target_needs_a_single_label_task():
    class WildcardLoss(BaseLoss):
        def forward(self, target: Tensor) -> Tensor: ...

    loss = WildcardLoss(node=DummyDetectionHead())
    with pytest.raises(RuntimeError, match="wildcard 'target' argument"):
        loss.get_parameters({}, {})


def test_missing_label_names_the_dataset_as_the_source():
    class TargetLoss(BaseLoss):
        def forward(self, target_segmentation: Tensor) -> Tensor: ...

    loss = TargetLoss(node=DummySegmentationHead())
    with pytest.raises(
        RuntimeError, match="but it is not present in the dataset"
    ):
        loss.get_parameters({}, {})


def test_missing_prediction_names_the_predictions_as_the_source():
    class PredictionLoss(BaseLoss):
        def forward(self, features: Tensor) -> Tensor: ...

    loss = PredictionLoss(node=DummySegmentationHead())
    with pytest.raises(
        RuntimeError, match="but it is not present in the predictions"
    ):
        loss.get_parameters({}, {})


def test_parameter_with_a_default_is_left_unbound():
    class DefaultLoss(BaseLoss):
        def forward(self, features: Tensor, scale: float = 1.0) -> Tensor: ...

    loss = DefaultLoss(node=DummySegmentationHead())
    assert list(loss.get_parameters({"features": torch.zeros(1)}, {})) == [
        "features"
    ]


def test_wrong_parameter_type_is_rejected():
    class TypedLoss(BaseLoss):
        def forward(self, features: Tensor) -> Tensor: ...

    loss = TypedLoss(node=DummySegmentationHead())
    with pytest.raises(TypeError, match="to be of type"):
        loss.get_parameters({"features": [torch.zeros(1)]}, {})
