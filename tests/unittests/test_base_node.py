import pytest
import torch
from torch import Size, Tensor

from luxonis_train.nodes import BaseNode
from luxonis_train.typing import AttachIndexType, Packet
from luxonis_train.utils import IncompatibleError


class DummyNode(BaseNode, register=False):
    def forward(self, _: Tensor) -> Tensor: ...


@pytest.fixture
def packet() -> Packet[Tensor]:
    return {"features": [torch.rand(3, 224, 224)]}


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
        ((-3, -1), [4, 5]),
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

    DummyNode.attach_index = 10
    with pytest.raises(ValueError, match="out of range"):
        BaseNode.get_attached(DummyNode, lst)  # type: ignore


def test_invalid(packet: Packet[Tensor]):
    node = DummyNode()
    with pytest.raises(RuntimeError, match="`input_shapes`"):
        _ = node.input_shapes
    with pytest.raises(RuntimeError, match="`original_in_shape`"):
        _ = node.original_in_shape
    with pytest.raises(RuntimeError, match="`dataset_metadata`"):
        _ = node.dataset_metadata


def test_in_sizes():
    DummyNode.attach_index = "all"
    node = DummyNode(
        input_shapes=[{"features": [Size((3, 224, 224)) for _ in range(3)]}]
    )
    assert node.in_sizes == [Size((3, 224, 224)) for _ in range(3)]
    node = DummyNode(in_sizes=Size((3, 224, 224)))
    assert node.in_sizes == Size((3, 224, 224))
    node = DummyNode(input_shapes=[{"feats": [Size((3, 224, 224))]}])
    assert node.in_sizes == [Size((3, 224, 224))]


def test_check_type_override():
    class DummyNode(BaseNode, register=False):
        in_channels: int
        attach_index = "all"

        def forward(self, _: Tensor) -> Tensor: ...

    with pytest.raises(IncompatibleError):
        DummyNode(
            input_shapes=[
                {"features": [Size((3, 224, 224)) for _ in range(3)]}
            ]
        )


def test_in_sizes_without_a_matching_signature_name():
    class Node(BaseNode, register=False):
        attach_index = "all"

        def forward(self, _: Tensor) -> Tensor: ...

    node = Node(input_shapes=[{"a": [Size((1,))], "b": [Size((2,))]}])
    with pytest.raises(RuntimeError, match="Unable to determine"):
        _ = node.in_sizes


def test_in_sizes_from_the_single_matching_signature_name():
    class Node(BaseNode, register=False):
        attach_index = "all"

        def forward(self, feats: Tensor) -> Tensor: ...

    node = Node(
        input_shapes=[{"feats": [Size((3, 8, 8))], "other": [Size((1,))]}]
    )
    assert node.in_sizes == [Size((3, 8, 8))]


def test_packet_parameter_receives_the_matching_input():
    class Node(BaseNode, register=False):
        def forward(self, a: Packet[Tensor]) -> Tensor:
            return a["features"][0]

    features = Node().run([{"features": [torch.zeros(2)]}])["features"]
    assert isinstance(features, Tensor)
    assert features.tolist() == [0.0, 0.0]


def test_packet_list_parameter_must_stand_alone():
    class Node(BaseNode, register=False):
        def forward(self, a: list[Packet[Tensor]], b: Tensor) -> Tensor: ...

    with pytest.raises(RuntimeError, match="not the only parameter"):
        Node().run([{"features": [torch.zeros(2)]}])


def test_too_few_input_packets_are_reported():
    class Node(BaseNode, register=False):
        def forward(self, a: Packet[Tensor], b: Packet[Tensor]) -> Tensor: ...

    with pytest.raises(RuntimeError, match="expects at least 2 inputs"):
        Node().run([{"features": [torch.zeros(2)]}])


def test_too_few_input_packets_for_tensors_are_reported():
    class Node(BaseNode, register=False):
        attach_index = -1

        def forward(self, x: Tensor, y: Tensor) -> Tensor: ...

    with pytest.raises(RuntimeError, match="expects at least 2 inputs"):
        Node().run([{"features": [torch.zeros(2)]}])


def test_unsupported_parameter_annotation_is_rejected():
    class Node(BaseNode, register=False):
        def forward(self, count: int) -> Tensor: ...

    with pytest.raises(TypeError, match="unsupported type"):
        Node().run([{"features": [torch.zeros(2)]}])


def test_numbered_parameter_selects_the_input_by_index():
    class Node(BaseNode, register=False):
        attach_index = -1

        def forward(self, input_1: Tensor) -> Tensor:
            return input_1

    features = Node().run(
        [{"features": [torch.zeros(2)]}, {"features": [torch.ones(2)]}]
    )["features"]
    assert isinstance(features, Tensor)
    assert features.tolist() == [1.0, 1.0]


def test_indexed_parameter_needs_the_features_key():
    class Node(BaseNode, register=False):
        def forward(self, x: Tensor) -> Tensor: ...

    with pytest.raises(RuntimeError, match="expects an input with key"):
        Node().run([{"other": [torch.zeros(2)]}])


def test_indexed_parameter_rejects_a_single_tensor_for_a_list():
    class Node(BaseNode, register=False):
        def forward(self, x: list[Tensor]) -> Tensor: ...

    with pytest.raises(RuntimeError, match="but got a single tensor"):
        Node().run([{"features": torch.zeros(2)}])


def test_non_standard_parameter_falls_back_to_the_only_input():
    class Node(BaseNode, register=False):
        attach_index = -1

        def forward(self, weird: Tensor) -> Tensor:
            return weird

    features = Node().run([{"embeddings": [torch.ones(2)]}])["features"]
    assert isinstance(features, Tensor)
    assert features.tolist() == [1.0, 1.0]
