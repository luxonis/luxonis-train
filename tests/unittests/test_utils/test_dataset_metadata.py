from typing import Any, cast

import pytest
from luxonis_ml.data import Category

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
    assert metadata.n_keypoints("segmentation") == 0
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


def test_dump_and_type_parsing() -> None:
    metadata = DatasetMetadata(
        classes={"task": {"cat": 0}},
        n_keypoints={"pose": 3},
        metadata_types=cast(
            Any,
            {
                "age": "int",
                "score": "float",
                "label": "str",
                "category": "Category",
            },
        ),
    )

    assert metadata.dump() == {
        "classes": {"task": {"cat": 0}},
        "n_keypoints": {"pose": 3},
        "metadata_types": {
            "age": "int",
            "score": "float",
            "label": "str",
            "category": "Category",
        },
    }
    assert metadata.metadata_types == {
        "age": int,
        "score": float,
        "label": str,
        "category": Category,
    }
    assert "classes" in str(metadata)
    assert repr(metadata) == str(metadata)
    assert dict(metadata.__rich_repr__()) == metadata.dump()

    with pytest.raises(ValueError, match="Unknown type name"):
        DatasetMetadata(metadata_types=cast(Any, {"bad": "unknown"}))


def test_metadata_types_missing_error(metadata: DatasetMetadata) -> None:
    object.__setattr__(metadata, "_metadata_types", None)
    with pytest.raises(RuntimeError, match="metadata types"):
        _ = metadata.metadata_types


def test_from_loader() -> None:
    class Loader:
        def get_classes(self) -> dict[str, dict[str, int]]:
            return {"task": {"cat": 0}}

        def get_n_keypoints(self) -> dict[str, int]:
            return {"pose": 5}

        def get_metadata_types(self) -> dict[str, type[str]]:
            return {"label": str}

    metadata = DatasetMetadata.from_loader(cast(Any, Loader()))

    assert metadata.classes("task") == {"cat": 0}
    assert metadata.n_keypoints("pose") == 5
    assert metadata.metadata_types == {"label": str}
