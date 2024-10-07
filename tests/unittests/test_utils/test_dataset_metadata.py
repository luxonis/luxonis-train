import pytest

from luxonis_train.utils import DatasetMetadata


@pytest.fixture
def metadata():
    return DatasetMetadata(
        classes={
            "color-segmentation": ["car", "person"],
            "detection": ["car", "person"],
        },
        n_keypoints={"color-segmentation": 0, "detection": 0},
    )


def test_n_classes(metadata):
    assert metadata.n_classes("color-segmentation") == 2
    assert metadata.n_classes("detection") == 2
    assert metadata.n_classes() == 2
    with pytest.raises(ValueError):
        metadata.n_classes("segmentation")
    metadata._classes["segmentation"] = ["car", "person", "tree"]
    with pytest.raises(RuntimeError):
        metadata.n_classes()


def test_n_keypoints(metadata):
    assert metadata.n_keypoints("color-segmentation") == 0
    assert metadata.n_keypoints("detection") == 0
    assert metadata.n_keypoints() == 0
    with pytest.raises(ValueError):
        metadata.n_keypoints("segmentation")
    metadata._n_keypoints["segmentation"] = 1
    with pytest.raises(RuntimeError):
        metadata.n_keypoints()


def test_class_names(metadata):
    assert metadata.classes("color-segmentation") == ["car", "person"]
    assert metadata.classes("detection") == ["car", "person"]
    assert metadata.classes() == ["car", "person"]
    with pytest.raises(ValueError):
        metadata.classes("segmentation")
    metadata._classes["segmentation"] = ["car", "person", "tree"]
    with pytest.raises(RuntimeError):
        metadata.classes()
