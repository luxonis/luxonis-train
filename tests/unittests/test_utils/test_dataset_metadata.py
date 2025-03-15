import pytest

from luxonis_train.utils import DatasetMetadata


@pytest.fixture
def metadata() -> DatasetMetadata:
    return DatasetMetadata(
        classes={
            "color-segmentation": {"car": 0, "person": 1},
            "detection": {"car": 0, "person": 1},
        },
        n_keypoints={"color-segmentation": 0, "detection": 0},
    )


def test_n_classes(metadata: DatasetMetadata):
    assert metadata.n_classes("color-segmentation") == 2
    assert metadata.n_classes("detection") == 2
    assert metadata.n_classes() == 2
    with pytest.raises(ValueError, match="Task 'segmentation'"):
        metadata.n_classes("segmentation")
    metadata._classes["segmentation"] = {"car": 0, "person": 1, "tree": 2}
    with pytest.raises(RuntimeError, match="different number of classes"):
        metadata.n_classes()


def test_n_keypoints(metadata: DatasetMetadata):
    assert metadata.n_keypoints("color-segmentation") == 0
    assert metadata.n_keypoints("detection") == 0
    assert metadata.n_keypoints() == 0
    with pytest.raises(ValueError, match="Task 'segmentation'"):
        metadata.n_keypoints("segmentation")
    metadata._n_keypoints["segmentation"] = 1
    with pytest.raises(RuntimeError, match="different number of keypoints"):
        metadata.n_keypoints()


def test_class_names(metadata: DatasetMetadata):
    assert metadata.classes("color-segmentation") == {"car": 0, "person": 1}
    assert metadata.classes("detection") == {"car": 0, "person": 1}
    assert metadata.classes() == {"car": 0, "person": 1}
    with pytest.raises(ValueError, match="Task 'segmentation'"):
        metadata.classes("segmentation")
    metadata._classes["segmentation"] = {"car": 0, "person": 1, "tree": 2}
    with pytest.raises(RuntimeError):
        metadata.classes()
