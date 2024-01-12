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
            {LabelType.CLASSIFICATION: torch.tensor([1, 0])},
        ),
        (
            torch.rand(3, 224, 224, dtype=torch.float32),
            {LabelType.CLASSIFICATION: torch.tensor([0, 1])},
        ),
    ]

    # Call collate_fn
    imgs, annotations = collate_fn(batch)

    # Check images tensor
    assert imgs.shape == (2, 3, 224, 224)
    assert imgs.dtype == torch.float32

    # Check annotations
    assert LabelType.CLASSIFICATION in annotations
    assert annotations[LabelType.CLASSIFICATION].shape == (2, 2)
    assert annotations[LabelType.CLASSIFICATION].dtype == torch.int64

    # TODO: test also segmentation, boundingbox and keypoint


if __name__ == "__main__":
    pytest.main()
