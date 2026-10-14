"""Training of multi-label classification and segmentation models."""

from pathlib import Path

import cv2
import numpy as np
import onnx
import pytest
from luxonis_ml.data import DatasetIterator
from luxonis_ml.typing import Params

from luxonis_train.core import LuxonisModel
from tests.conftest import LuxonisTestDataset

SIZE = 64


@pytest.fixture
def overlapping_squares(data_dir: Path) -> LuxonisTestDataset:
    """Build images with a red and a blue square.

    Each square is a class of the image and the mask of its pixels. In
    half of the images, the squares overlap, so a pixel can be red and
    blue. No square covers the corners, so the loader adds the class
    ``background``.

    """
    data_dir = data_dir / "overlapping_squares"
    data_dir.mkdir(parents=True, exist_ok=True)

    def generator() -> DatasetIterator:
        for i in range(8):
            image = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
            offset = 16 if i % 2 else 32
            squares = {
                "red": (slice(8, 40), slice(8, 40)),
                "blue": (
                    slice(offset, offset + 24),
                    slice(offset, offset + 24),
                ),
            }
            path = data_dir / f"squares_{i}.png"
            for class_name, (rows, cols) in squares.items():
                image[rows, cols, 2 if class_name == "red" else 0] = 255
            cv2.imwrite(str(path), image)
            for class_name, (rows, cols) in squares.items():
                mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
                mask[rows, cols] = 1
                yield {
                    "file": str(path),
                    "annotation": {
                        "class": class_name,
                        "segmentation": {"mask": mask},
                    },
                }

    dataset = LuxonisTestDataset(
        "overlapping_squares", delete_local=True, source_path=data_dir
    )
    dataset.add(generator())
    dataset.make_splits((0.5, 0.25, 0.25))
    return dataset


@pytest.mark.parametrize(
    ("model_name", "export_shape"),
    [
        ("ClassificationModel", [1, 3]),
        ("SegmentationModel", [1, 3, 32, 64]),
    ],
)
def test_multilabel_training(
    model_name: str,
    export_shape: list[int],
    overlapping_squares: LuxonisTestDataset,
    opts: Params,
):
    model = LuxonisModel(
        {
            "model": {
                "name": model_name,
                "predefined_model": {
                    "name": model_name,
                    "params": {"variant": "light", "multilabel": True},
                },
            },
            "loader": {
                "params": {"dataset_name": overlapping_squares.identifier}
            },
        },
        opts,
    )

    model.train()
    model.test()
    model.export()

    graph = onnx.load(
        str(model.run_save_dir / "export" / f"{model_name}.onnx")
    ).graph
    dims = graph.output[0].type.tensor_type.shape.dim
    assert [dim.dim_value for dim in dims] == export_shape
