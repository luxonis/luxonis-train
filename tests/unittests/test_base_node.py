import pytest
import torch
from torch import Size, Tensor

from luxonis_train.enums import TaskType
from luxonis_train.nodes import AttachIndexType, BaseNode
from luxonis_train.utils import DatasetMetadata, Packet
from luxonis_train.utils.exceptions import IncompatibleException


class DummyNode(BaseNode, register=False):
    def forward(self, _): ...


@pytest.fixture
def packet() -> Packet[Tensor]:
    return {
        "features": [torch.rand(3, 224, 224)],
    }


@pytest.mark.parametrize(
    ("attach_index", "expected"),
    [
        (-1, 5),
        (0, 1),
        ("all", [1, 2, 3, 4, 5]),
        ((0, 2), [1, 2]),
        ((0, 4, 2), [1, 3]),
        ((-1, -3, -1), [5, 4]),
        ((4, 2), [5, 4]),
        ((-1, -3), [5, 4]),
        ((-4, 4), [2, 3, 4]),
        ((1, -1), [2, 3, 4]),
    ],
)
def test_attach_index(
    attach_index: AttachIndexType, expected: list[int] | int
):
    lst = [1, 2, 3, 4, 5]

    class DummyBaseNode:
        attach_index: AttachIndexType

    DummyBaseNode.attach_index = attach_index

    assert BaseNode.get_attached(DummyBaseNode, lst) == expected  # type: ignore


def test_attach_index_error():
    lst = [1, 2, 3, 4, 5]

    class DummyNode(BaseNode, register=False):
        attach_index: AttachIndexType

    with pytest.raises(ValueError):
        DummyNode.attach_index = 10
        BaseNode.get_attached(DummyNode, lst)  # type: ignore

    with pytest.raises(ValueError):
        DummyNode.attach_index = "none"  # type: ignore
        BaseNode.get_attached(DummyNode, lst)  # type: ignore


def test_invalid(packet: Packet[Tensor]):
    node = DummyNode()
    with pytest.raises(RuntimeError):
        _ = node.input_shapes
    with pytest.raises(RuntimeError):
        _ = node.original_in_shape
    with pytest.raises(RuntimeError):
        _ = node.dataset_metadata
    with pytest.raises(ValueError):
        node.unwrap([packet, packet])
    with pytest.raises(ValueError):
        node.wrap({"inp": torch.rand(3, 224, 224)})


def test_in_sizes():
    node = DummyNode(
        input_shapes=[{"features": [Size((3, 224, 224)) for _ in range(3)]}]
    )
    assert node.in_sizes == [Size((3, 224, 224)) for _ in range(3)]
    node = DummyNode(in_sizes=Size((3, 224, 224)))
    assert node.in_sizes == Size((3, 224, 224))
    with pytest.raises(RuntimeError):
        node = DummyNode(input_shapes=[{"feats": [Size((3, 224, 224))]}])
        _ = node.in_sizes


def test_check_type_override():
    class DummyNode(BaseNode, register=False):
        in_channels: int

        def forward(self, _): ...

    with pytest.raises(IncompatibleException):
        DummyNode(
            input_shapes=[
                {"features": [Size((3, 224, 224)) for _ in range(3)]}
            ]
        )


def test_tasks():
    class DummyHead(DummyNode):
        tasks = [TaskType.CLASSIFICATION]

    class DummyMultiHead(DummyNode):
        tasks = [TaskType.CLASSIFICATION, TaskType.SEGMENTATION]

    dummy_head = DummyHead()
    dummy_node = DummyNode()
    dummy_multi_head = DummyMultiHead(n_keypoints=4)
    assert (
        dummy_head.get_task_name(TaskType.CLASSIFICATION) == "classification"
    )
    assert dummy_head.task == "classification"
    with pytest.raises(ValueError):
        dummy_head.get_task_name(TaskType.SEGMENTATION)

    with pytest.raises(RuntimeError):
        dummy_node.get_task_name(TaskType.SEGMENTATION)

    with pytest.raises(RuntimeError):
        _ = dummy_node.task

    with pytest.raises(ValueError):
        _ = dummy_multi_head.task

    metadata = DatasetMetadata(
        classes={
            "segmentation": ["car", "person", "dog"],
            "classification": ["car-class", "person-class"],
        },
        n_keypoints={"color-segmentation": 0, "detection": 0},
    )

    dummy_multi_head._dataset_metadata = metadata
    assert dummy_multi_head.get_class_names(TaskType.SEGMENTATION) == [
        "car",
        "person",
        "dog",
    ]
    assert dummy_multi_head.get_class_names(TaskType.CLASSIFICATION) == [
        "car-class",
        "person-class",
    ]
    assert dummy_multi_head.get_n_classes(TaskType.SEGMENTATION) == 3
    assert dummy_multi_head.get_n_classes(TaskType.CLASSIFICATION) == 2
    assert dummy_multi_head.n_keypoints == 4
    with pytest.raises(ValueError):
        _ = dummy_head.n_keypoints
    with pytest.raises(RuntimeError):
        _ = dummy_node.n_keypoints

    dummy_head = DummyHead(n_classes=5)
    assert dummy_head.n_classes == 5
    with pytest.raises(ValueError):
        _ = dummy_multi_head.n_classes
