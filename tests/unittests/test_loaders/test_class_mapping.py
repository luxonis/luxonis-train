from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from luxonis_train.attached_modules.losses import CrossEntropyLoss
from luxonis_train.core import LuxonisModel
from luxonis_train.loaders import LuxonisLoaderTorch
from luxonis_train.utils.dataset_metadata import DatasetMetadata


def _loader(source: dict[str, dict[str, int]]) -> LuxonisLoaderTorch:
    loader = LuxonisLoaderTorch.__new__(LuxonisLoaderTorch)
    loader.loader = cast(Any, SimpleNamespace(_classes=source))
    loader._class_mapping = {}
    loader._target_classes = None
    return loader


@pytest.mark.parametrize("extra_background_view", ["val", "test"])
def test_metadata_includes_background_required_by_another_view(
    extra_background_view: str, monkeypatch: pytest.MonkeyPatch
):
    """Reserve a real output for background before constructing a new model."""
    source = {"objects": {"cat": 0, "dog": 1}}
    loaders = {view: _loader(source) for view in ("train", "val", "test")}
    loaders[extra_background_view] = _loader(
        {"objects": {"background": 0, "dog": 1, "cat": 2}}
    )
    monkeypatch.setattr(LuxonisLoaderTorch, "get_n_keypoints", lambda _: {})
    monkeypatch.setattr(LuxonisLoaderTorch, "get_metadata_types", lambda _: {})
    model = LuxonisModel.__new__(LuxonisModel)
    model.loaders = cast(Any, loaders)
    model._dataset_metadata = model._resolve_dataset_metadata(None)
    model._align_loader_classes()
    target = {"objects": {"background": 0, "cat": 1, "dog": 2}}
    assert model._dataset_metadata.dump()["classes"] == target
    assert all(loader.get_classes() == target for loader in loaders.values())
    assert loaders["train"].loader._classes == source

    mask = np.array([[[1, 0, 0]], [[0, 1, 0]], [[0, 0, 1]]])
    remapped = loaders[extra_background_view]._remap_classes(
        {"objects/segmentation": mask}
    )["objects/segmentation"]
    np.testing.assert_array_equal(remapped, mask[[0, 2, 1]])
    np.testing.assert_array_equal(remapped.argmax(axis=0), [[0, 2, 1]])
    train_mask = np.array([[[1, 0, 0]], [[0, 1, 0]]])
    remapped_train = loaders["train"]._remap_classes(
        {"objects/segmentation": train_mask}
    )["objects/segmentation"]
    np.testing.assert_array_equal(remapped_train.argmax(axis=0), [[1, 2, 0]])


def test_fixed_metadata_does_not_silently_gain_background():
    """Keep checkpoint output counts fixed and reject unrepresentable labels."""
    metadata = DatasetMetadata(classes={"objects": {"cat": 0, "dog": 1}})
    model = LuxonisModel.__new__(LuxonisModel)
    model.loaders = cast(
        Any,
        {
            "train": _loader({"objects": {"cat": 0, "dog": 1}}),
            "val": _loader({"objects": {"background": 0, "cat": 1, "dog": 2}}),
        },
    )
    model._dataset_metadata = model._resolve_dataset_metadata(metadata)
    assert model._dataset_metadata is metadata
    with pytest.raises(
        ValueError, match=r"background.*absent from model classes"
    ):
        model._align_loader_classes()
    assert metadata.dump()["classes"] == {"objects": {"cat": 0, "dog": 1}}


@pytest.mark.parametrize("from_path", [False, True])
@pytest.mark.parametrize("different_classes", [False, True])
def test_test_weights_class_contract(
    from_path: bool,
    different_classes: bool,
    tmp_path: Path,
):
    """Check class contracts before loading weights or starting evaluation."""
    model = LuxonisModel.__new__(LuxonisModel)
    model._weights = None
    model._weights_provided_in_config = False
    model._weights_provided_during_init = False
    model._dataset_metadata = DatasetMetadata(
        classes={"objects": {"cat": 0, "dog": 1}}
    )
    classes = (
        {"dog": 0, "cat": 1} if different_classes else {"cat": 0, "dog": 1}
    )
    checkpoint = {
        "state_dict": {},
        "dataset_metadata": {"classes": {"objects": classes}},
    }
    weights: Path | dict[str, Any] = checkpoint
    if from_path:
        weights = tmp_path / "test.ckpt"
        torch.save(checkpoint, weights)
    model.pytorch_loaders = cast(Any, {"test": object()})
    module = Mock(state_dict=Mock(return_value={}))
    trainer = Mock()
    trainer.test.return_value = [{"test/metric": 1.0}]
    model.lightning_module = cast(Any, module)
    model.pl_trainer = trainer
    if different_classes:
        with pytest.raises(ValueError, match="checkpoint classes differ"):
            model.test(weights=weights, finalize_tracker=False)
        module.load_checkpoint.assert_not_called()
        trainer.test.assert_not_called()
    else:
        assert model.test(weights=weights, finalize_tracker=False) == {
            "test/metric": 1.0
        }
        module.load_checkpoint.assert_called_once_with(checkpoint)
        module.load_state_dict.assert_called_once_with({})


def test_test_weights_accept_legacy_and_bare_state_dicts():
    """Keep checkpoints without class metadata usable for evaluation."""
    model = LuxonisModel.__new__(LuxonisModel)
    state_dict = {"weight": torch.ones(1)}
    assert model._resolve_test_weights(None) is None
    assert model._resolve_test_weights(state_dict) == {
        "state_dict": state_dict
    }
    assert model._resolve_test_weights({"state_dict": state_dict}) == {
        "state_dict": state_dict
    }


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


@pytest.mark.parametrize("background_index", [0, 1])
@pytest.mark.parametrize("empty", [False, True])
def test_segmentation_mapping_preserves_implicit_background(
    background_index: int, empty: bool
):
    loader = _loader({"objects": {"cat": 0}})
    loader.set_class_mapping(
        {
            "objects": {
                "cat": 1 - background_index,
                "background": background_index,
            }
        }
    )
    mask = np.array([[[1, 0], [0, 1]]], dtype=np.float32)
    if empty:
        mask.fill(0)
    original = mask.copy()
    remapped = loader._remap_classes({"objects/segmentation": mask})[
        "objects/segmentation"
    ]
    expected = np.empty_like(remapped)
    expected[background_index] = 1 - mask[0]
    expected[1 - background_index] = mask[0]
    np.testing.assert_array_equal(remapped, expected)
    np.testing.assert_array_equal(mask, original)
    target = torch.from_numpy(expected).unsqueeze(0)
    predictions = target * 20 - 10
    torch.testing.assert_close(
        CrossEntropyLoss()(
            predictions, torch.from_numpy(remapped).unsqueeze(0)
        ),
        torch.tensor(0.0),
    )


def test_classification_mapping_does_not_invent_background_labels():
    loader = _loader({"objects": {"cat": 0}})
    loader.set_class_mapping({"objects": {"cat": 0, "background": 1}})
    remapped = loader._remap_classes({"objects/classification": np.array([0])})
    np.testing.assert_array_equal(remapped["objects/classification"], [0, 0])


def test_segmentation_mapping_preserves_explicit_background():
    loader = _loader({"objects": {"background": 0, "cat": 1}})
    loader.set_class_mapping({"objects": {"cat": 0, "background": 1}})
    mask = np.array([[[1, 0, 0]], [[0, 1, 0]]], dtype=np.float32)
    remapped = loader._remap_classes({"objects/segmentation": mask})
    np.testing.assert_array_equal(
        remapped["objects/segmentation"], mask[[1, 0]]
    )
