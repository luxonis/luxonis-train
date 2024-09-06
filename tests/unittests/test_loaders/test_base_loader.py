import pytest
import torch
from luxonis_ml.data import LabelType

from luxonis_train.loaders import collate_fn


@pytest.mark.parametrize(
    "input_names_and_shapes",
    [
        [("features", torch.Size([3, 224, 224]))],
        [
            ("features", torch.Size([3, 224, 224])),
            ("segmentation", torch.Size([1, 224, 224])),
        ],
        [
            ("features", torch.Size([3, 224, 224])),
            ("segmentation", torch.Size([1, 224, 224])),
            ("disparity", torch.Size([1, 224, 224])),
        ],
        [
            ("features", torch.Size([3, 224, 224])),
            ("pointcloud", torch.Size([1000, 3])),
        ],
        [
            ("features", torch.Size([3, 224, 224])),
            ("pointcloud", torch.Size([1000, 3])),
            ("foobar", torch.Size([2, 3, 4, 5, 6])),
        ],
    ],
)
@pytest.mark.parametrize("batch_size", [1, 2])
def test_collate_fn(input_names_and_shapes, batch_size):
    # Mock batch data

    def build_batch_element():
        inputs = {}
        for name, shape in input_names_and_shapes:
            inputs[name] = torch.rand(shape, dtype=torch.float32)

        labels = {
            "classification": (
                torch.randint(0, 2, (2,), dtype=torch.int64),
                LabelType.CLASSIFICATION,
            )
        }

        return inputs, labels

    batch = [build_batch_element() for _ in range(batch_size)]

    # Call collate_fn
    inputs, annotations = collate_fn(batch)  # type: ignore

    # Check images tensor
    assert inputs["features"].shape == (batch_size, 3, 224, 224)
    assert inputs["features"].dtype == torch.float32

    # Check annotations
    assert "classification" in annotations
    assert annotations["classification"][0].shape == (batch_size, 2)
    assert annotations["classification"][0].dtype == torch.int64


# TODO: test also segmentation, boundingbox and keypoint


if __name__ == "__main__":
    pytest.main()
