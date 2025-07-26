import pytest
import torch
from pytest_subtests import SubTests
from torch import Size

from luxonis_train.loaders import BaseLoaderTorch, LuxonisLoaderTorchOutput


class DummyLoader(BaseLoaderTorch):
    def __len__(self) -> int: ...

    def get(self, idx: int) -> LuxonisLoaderTorchOutput: ...

    def get_classes(self) -> dict[str, dict[str, int]]: ...

    def input_shapes(self) -> dict[str, Size]: ...


@pytest.mark.parametrize(
    "input_names_and_shapes",
    [
        [("features", Size([3, 224, 224]))],
        [
            ("features", Size([3, 224, 224])),
            ("segmentation", Size([1, 224, 224])),
        ],
        [
            ("features", Size([3, 224, 224])),
            ("segmentation", Size([1, 224, 224])),
            ("disparity", Size([1, 224, 224])),
        ],
        [
            ("features", Size([3, 224, 224])),
            ("pointcloud", Size([1000, 3])),
        ],
        [
            ("features", Size([3, 224, 224])),
            ("pointcloud", Size([1000, 3])),
            ("foobar", Size([2, 3, 4, 5, 6])),
        ],
    ],
)
@pytest.mark.parametrize("batch_size", [1, 2])
def test_collate_fn(
    input_names_and_shapes: list[tuple[str, Size]],
    batch_size: int,
    subtests: SubTests,
):
    def build_batch_element() -> LuxonisLoaderTorchOutput:
        inputs = {}
        for name, shape in input_names_and_shapes:
            inputs[name] = torch.rand(shape, dtype=torch.float32)

        labels = {
            "/classification": (torch.randint(0, 2, (2,), dtype=torch.int64)),
            "/segmentation": (
                torch.randint(0, 2, (1, 224, 224), dtype=torch.int64)
            ),
            "/keypoints": (torch.rand(1, 52, dtype=torch.float32)),
            "/boundingbox": (torch.rand(1, 5, dtype=torch.float32)),
        }

        return inputs, labels

    batch = [build_batch_element() for _ in range(batch_size)]

    loader = DummyLoader(view=["train"])
    inputs, annotations = loader.collate_fn(batch)

    with subtests.test("inputs"):
        assert inputs["features"].shape == (batch_size, 3, 224, 224)
        assert inputs["features"].dtype == torch.float32

    with subtests.test("classification"):
        assert "/classification" in annotations
        assert annotations["/classification"].shape == (batch_size, 2)
        assert annotations["/classification"].dtype == torch.int64

    with subtests.test("segmentation"):
        assert "/segmentation" in annotations
        assert annotations["/segmentation"].shape == (batch_size, 1, 224, 224)
        assert annotations["/segmentation"].dtype == torch.int64

    with subtests.test("keypoints"):
        assert "/keypoints" in annotations
        assert annotations["/keypoints"].shape == (batch_size, 53)
        assert annotations["/keypoints"].dtype == torch.float32

    with subtests.test("boundingbox"):
        assert "/boundingbox" in annotations
        assert annotations["/boundingbox"].shape == (batch_size, 6)
        assert annotations["/boundingbox"].dtype == torch.float32
