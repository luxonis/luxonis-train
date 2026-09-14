import pytest
import torch
from pytest_subtests import SubTests
from torch import Size

from luxonis_train.loaders import (
    BaseLoaderTorch,
    LuxonisLoaderTorch,
    LuxonisLoaderTorchOutput,
)


class DummyLoader(BaseLoaderTorch):
    def __len__(self) -> int: ...

    def __getitem__(self, idx: int) -> LuxonisLoaderTorchOutput: ...

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
        assert isinstance(inputs, dict)
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


@pytest.mark.parametrize("tensor_first", [True, False])
def test_collate_fn_rejects_mixed_input_types(tensor_first: bool):
    loader = DummyLoader(view=["train"])
    labels = {"/classification": torch.randint(0, 2, (2,), dtype=torch.int64)}
    tensor_sample = (torch.rand(3, 8, 8, dtype=torch.float32), labels)
    dict_sample = (
        {"features": torch.rand(3, 8, 8, dtype=torch.float32)},
        labels,
    )
    batch = (
        [tensor_sample, dict_sample]
        if tensor_first
        else [dict_sample, tensor_sample]
    )

    with pytest.raises(TypeError, match="same input type"):
        loader.collate_fn(batch)


def test_keypoint_mapping_rejects_an_unknown_task():
    with pytest.raises(KeyError, match="not present in dataset tasks"):
        LuxonisLoaderTorch._validate_keypoint_task(
            "hands", {"faces": ["keypoints"]}
        )


def test_keypoint_mapping_rejects_a_task_without_keypoints():
    with pytest.raises(KeyError, match="doesn't have `keypoints`"):
        LuxonisLoaderTorch._validate_keypoint_task(
            "faces", {"faces": ["boundingbox"]}
        )
