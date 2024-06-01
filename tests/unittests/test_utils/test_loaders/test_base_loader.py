import pytest
import torch

from luxonis_train.utils.loaders import (
    collate_fn,
)
from luxonis_train.utils.types import LabelType


def test_collate_fn():
    # Mock batch data
    batch = [
        (
            torch.rand(3, 224, 224, dtype=torch.float32),
            {"classification": (torch.tensor([1, 0]), LabelType.CLASSIFICATION)},
        ),
        (
            torch.rand(3, 224, 224, dtype=torch.float32),
            {"classification": (torch.tensor([0, 1]), LabelType.CLASSIFICATION)},
        ),
    ]

    # Call collate_fn
    imgs, annotations = collate_fn(batch)  # type: ignore

    # Check images tensor
    assert imgs.shape == (2, 3, 224, 224)
    assert imgs.dtype == torch.float32

    # Check annotations
    assert "classification" in annotations
    assert annotations["classification"][0].shape == (2, 2)
    assert annotations["classification"][0].dtype == torch.int64

    # TODO: test also segmentation, boundingbox and keypoint


if __name__ == "__main__":
    pytest.main()
