import pytest
import torch
from torch import Size

from luxonis_train.enums import TaskType
from luxonis_train.loaders import collate_fn


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
    input_names_and_shapes: list[tuple[str, Size]], batch_size: int, subtests
):
    def build_batch_element():
        inputs = {}
        for name, shape in input_names_and_shapes:
            inputs[name] = torch.rand(shape, dtype=torch.float32)

        labels = {
            "classification": (
                torch.randint(0, 2, (2,), dtype=torch.int64),
                TaskType.CLASSIFICATION,
            ),
            "segmentation": (
                torch.randint(0, 2, (1, 224, 224), dtype=torch.int64),
                TaskType.SEGMENTATION,
            ),
            "keypoints": (
                torch.rand(1, 52, dtype=torch.float32),
                TaskType.KEYPOINTS,
            ),
            "boundingbox": (
                torch.rand(1, 5, dtype=torch.float32),
                TaskType.BOUNDINGBOX,
            ),
        }

        return inputs, labels

    batch = [build_batch_element() for _ in range(batch_size)]

    inputs, annotations = collate_fn(batch)  # type: ignore

    with subtests.test("inputs"):
        assert inputs["features"].shape == (batch_size, 3, 224, 224)
        assert inputs["features"].dtype == torch.float32

    with subtests.test("classification"):
        assert "classification" in annotations
        assert annotations["classification"][0].shape == (batch_size, 2)
        assert annotations["classification"][0].dtype == torch.int64

    with subtests.test("segmentation"):
        assert "segmentation" in annotations
        assert annotations["segmentation"][0].shape == (
            batch_size,
            1,
            224,
            224,
        )
        assert annotations["segmentation"][0].dtype == torch.int64

    with subtests.test("keypoints"):
        assert "keypoints" in annotations
        assert annotations["keypoints"][0].shape == (batch_size, 53)
        assert annotations["keypoints"][0].dtype == torch.float32

    with subtests.test("boundingbox"):
        assert "boundingbox" in annotations
        assert annotations["boundingbox"][0].shape == (batch_size, 6)
        assert annotations["boundingbox"][0].dtype == torch.float32
