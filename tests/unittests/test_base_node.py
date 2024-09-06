import pytest

from luxonis_train.nodes import AttachIndexType, BaseNode


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
def test_attach_index(attach_index: AttachIndexType, expected: list[int] | int):
    lst = [1, 2, 3, 4, 5]

    class DummyBaseNode:
        attach_index: AttachIndexType

    DummyBaseNode.attach_index = attach_index

    assert BaseNode.get_attached(DummyBaseNode, lst) == expected  # type: ignore


def test_attach_index_error():
    lst = [1, 2, 3, 4, 5]

    class DummyBaseNode:
        attach_index: AttachIndexType

    with pytest.raises(ValueError):
        DummyBaseNode.attach_index = 10
        BaseNode.get_attached(DummyBaseNode, lst)  # type: ignore

    with pytest.raises(ValueError):
        DummyBaseNode.attach_index = "none"  # type: ignore
        BaseNode.get_attached(DummyBaseNode, lst)  # type: ignore
