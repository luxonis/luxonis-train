import pytest
from luxonis_ml.data import LabelType

from luxonis_train import BaseLoss, BaseNode
from luxonis_train.utils.exceptions import IncompatibleException


class DummyBackbone(BaseNode):
    def forward(self, _): ...


class DummySegmentationHead(BaseNode):
    tasks = [LabelType.SEGMENTATION]

    def forward(self, _): ...


class DummyBBoxHead(BaseNode):
    tasks = [LabelType.BOUNDINGBOX]

    def forward(self, _): ...


class DummyDetectionHead(BaseNode):
    tasks = [LabelType.BOUNDINGBOX, LabelType.KEYPOINTS]

    def forward(self, _): ...


class DummyLoss(BaseLoss):
    supported_labels = [
        LabelType.SEGMENTATION,
        (LabelType.KEYPOINTS, LabelType.BOUNDINGBOX),
    ]

    def forward(self, _): ...


class NoLabelLoss(BaseLoss):
    def forward(self, _): ...


@pytest.fixture
def labels():
    return {
        "segmentation": ("segmentation", LabelType.SEGMENTATION),
        "keypoints": ("keypoints", LabelType.KEYPOINTS),
        "boundingbox": ("boundingbox", LabelType.BOUNDINGBOX),
        "classification": ("classification", LabelType.CLASSIFICATION),
    }


@pytest.fixture
def inputs():
    return {
        "features": ["features"],
        "segmentation": ["segmentation"],
    }


def test_valid_properties():
    head = DummySegmentationHead()
    loss = DummyLoss(node=head)
    no_labels_loss = NoLabelLoss(node=head)
    assert loss.node == head
    assert loss.node_tasks == {LabelType.SEGMENTATION: "segmentation"}
    assert loss.required_labels == [LabelType.SEGMENTATION]
    assert no_labels_loss.node == head
    assert no_labels_loss.node_tasks == {
        LabelType.SEGMENTATION: "segmentation"
    }
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


def test_get_label(labels):
    seg_head = DummySegmentationHead()
    det_head = DummyDetectionHead()
    seg_loss = DummyLoss(node=seg_head)
    assert seg_loss.get_label(labels) == "segmentation"
    assert seg_loss.get_label(labels, LabelType.SEGMENTATION) == "segmentation"

    del labels["segmentation"]
    labels["segmentation-task"] = ("segmentation", LabelType.SEGMENTATION)

    with pytest.raises(IncompatibleException):
        seg_loss.get_label(labels)

    det_loss = DummyLoss(node=det_head)
    assert det_loss.get_label(labels, LabelType.KEYPOINTS) == "keypoints"
    assert det_loss.get_label(labels, LabelType.BOUNDINGBOX) == "boundingbox"

    with pytest.raises(ValueError):
        det_loss.get_label(labels)

    with pytest.raises(ValueError):
        det_loss.get_label(labels, LabelType.SEGMENTATION)


def test_input_tensors(inputs):
    seg_head = DummySegmentationHead()
    seg_loss = DummyLoss(node=seg_head)
    assert seg_loss.get_input_tensors(inputs) == ["segmentation"]
    assert seg_loss.get_input_tensors(inputs, "segmentation") == [
        "segmentation"
    ]
    assert seg_loss.get_input_tensors(inputs, LabelType.SEGMENTATION) == [
        "segmentation"
    ]

    with pytest.raises(IncompatibleException):
        seg_loss.get_input_tensors(inputs, LabelType.KEYPOINTS)
    with pytest.raises(IncompatibleException):
        seg_loss.get_input_tensors(inputs, "keypoints")

    det_head = DummyDetectionHead()
    det_loss = DummyLoss(node=det_head)
    with pytest.raises(ValueError):
        det_loss.get_input_tensors(inputs)


def test_prepare(inputs, labels):
    backbone = DummyBackbone()
    seg_head = DummySegmentationHead()
    seg_loss = DummyLoss(node=seg_head)
    det_head = DummyDetectionHead()

    assert seg_loss.prepare(inputs, labels) == ("segmentation", "segmentation")
    inputs["segmentation"].append("segmentation2")
    assert seg_loss.prepare(inputs, labels) == (
        "segmentation2",
        "segmentation",
    )

    with pytest.raises(RuntimeError):
        NoLabelLoss(node=backbone).prepare(inputs, labels)

    with pytest.raises(RuntimeError):
        NoLabelLoss(node=seg_head).prepare(inputs, labels)

    with pytest.raises(RuntimeError):
        DummyLoss(node=det_head).prepare(inputs, labels)
