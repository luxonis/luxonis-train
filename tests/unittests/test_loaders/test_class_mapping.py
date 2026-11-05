from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch

from luxonis_train.core import LuxonisModel
from luxonis_train.loaders import LuxonisLoaderTorch
from luxonis_train.utils.dataset_metadata import DatasetMetadata


def _loader(source: dict[str, dict[str, int]]) -> LuxonisLoaderTorch:
    loader = LuxonisLoaderTorch.__new__(LuxonisLoaderTorch)
    loader.loader = cast(Any, SimpleNamespace(_classes=source))
    loader._class_mapping = {}
    loader._target_classes = None
    return loader


def test_align_loader_labels_with_checkpoint_classes():
    source = {"objects": {"cat": 0, "dog": 1}}
    target = {"objects": {"dog": 0, "absent": 1, "cat": 2}}
    loaders = {view: _loader(source) for view in ("train", "val", "test")}
    model = cast(
        Any,
        SimpleNamespace(
            loaders=loaders, _dataset_metadata=DatasetMetadata(classes=target)
        ),
    )
    LuxonisModel._align_loader_classes(model)
    labels = {
        "objects/boundingbox": np.array(
            [[0, 0.1, 0.2, 0.3, 0.4], [1, 0.2, 0.3, 0.4, 0.5]]
        ),
        "objects/classification": np.array([1, 0]),
        "objects/segmentation": np.arange(8).reshape(2, 2, 2),
        "objects/keypoints": np.ones((2, 12)),
        "objects/instance_segmentation": np.ones((2, 2, 2)),
    }
    original = {key: value.copy() for key, value in labels.items()}
    for loader in loaders.values():
        remapped = loader._remap_classes(labels)
        assert loader.get_classes() == target
        assert loader.loader._classes == source
        np.testing.assert_array_equal(
            remapped["objects/boundingbox"][:, 0], [2, 0]
        )
        np.testing.assert_array_equal(
            remapped["objects/classification"], [0, 0, 1]
        )
        np.testing.assert_array_equal(
            remapped["objects/segmentation"][[2, 0]],
            labels["objects/segmentation"],
        )
        assert not remapped["objects/segmentation"][1].any()
        for key in ("objects/keypoints", "objects/instance_segmentation"):
            np.testing.assert_array_equal(remapped[key], labels[key])
        for key, value in labels.items():
            np.testing.assert_array_equal(value, original[key])
        empty = loader._remap_classes(
            {"objects/boundingbox": np.empty((0, 5))}
        )
        assert empty["objects/boundingbox"].shape == (0, 5)
        loader.set_class_mapping(source)
        assert loader._remap_classes(labels) is labels


@pytest.mark.parametrize(
    ("target", "message"),
    [
        ({}, "absent from model classes"),
        ({"objects": {"dog": 0}}, "absent from model classes"),
        ({"objects": {"cat": 1}}, "contiguous from zero"),
    ],
)
def test_class_mapping_rejects_incompatible_metadata(
    target: dict[str, dict[str, int]], message: str
):
    loader = _loader({"objects": {"cat": 0}})
    with pytest.raises(ValueError, match=message):
        loader.set_class_mapping(target)


def test_getitem_emits_remapped_labels():
    class AugmentedLoader:
        _classes = {"objects": {"cat": 0}}

        def __getitem__(
            self, index: int
        ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
            return np.zeros((2, 2, 3), dtype=np.uint8), {
                "objects/boundingbox": np.array([[0, 0.1, 0.2, 0.3, 0.4]]),
                "objects/classification": np.array([1]),
            }

    loader = _loader(AugmentedLoader._classes)
    loader.loader = cast(Any, AugmentedLoader())
    loader._image_source = "image"
    loader.kpts_mapping_per_task = None
    loader._return_sample_metadata = False
    loader.set_class_mapping({"objects": {"dog": 0, "cat": 1}})
    output = loader[0]
    image, labels = output[0], output[1]
    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, 2, 2)
    torch.testing.assert_close(
        labels["objects/classification"], torch.tensor([0.0, 1.0])
    )
    assert labels["objects/boundingbox"][0, 0] == 1
