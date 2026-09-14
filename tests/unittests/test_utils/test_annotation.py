from pathlib import Path
from typing import Any
from unittest.mock import Mock

import cv2
import numpy as np
import pytest
import torch
from pydantic import ValidationError
from torch import Tensor

from luxonis_train.config.config import PreprocessingConfig
from luxonis_train.core.utils import annotate_utils
from luxonis_train.core.utils.annotate_utils import _annotated_records
from luxonis_train.nodes import BaseHead
from luxonis_train.tasks import Tasks
from luxonis_train.typing import Packet
from luxonis_train.utils import DatasetMetadata
from luxonis_train.utils.annotation import default_annotate


class KeypointHead(BaseHead, register=False):
    task = Tasks.KEYPOINTS

    def forward(self, _: Tensor) -> Tensor: ...


class EmbeddingHead(BaseHead, register=False):
    task = Tasks.EMBEDDINGS

    def forward(self, _: Tensor) -> Tensor: ...


class ClassificationHead(BaseHead, register=False):
    task = Tasks.CLASSIFICATION

    def forward(self, _: Tensor) -> Tensor: ...


class OCRHead(BaseHead, register=False):
    task = Tasks.OCR

    def forward(self, _: Tensor) -> Tensor: ...

    def decoder(self, predictions: Tensor) -> list[tuple[str, float]]:
        assert predictions.shape == (1, 1, 1)
        return [(str(int(predictions.item())), 1.0)]


def _write_image(path: Path) -> None:
    cv2.imwrite(str(path), np.zeros((100, 200, 3), dtype=np.uint8))


def _annotation(record: object) -> dict[str, Any]:
    assert isinstance(record, dict)
    annotation = record["annotation"]
    assert isinstance(annotation, dict)
    return annotation


def test_unsupported_task_is_rejected():
    with pytest.raises(ValueError, match="Unsupported task"):
        list(default_annotate(EmbeddingHead(), {}, [], PreprocessingConfig()))


def test_keypoints_are_normalized_to_the_original_image(tmp_path: Path):
    image = tmp_path / "image.png"
    _write_image(image)
    head_output: Packet[Tensor] = {
        "keypoints": torch.tensor([[[[50.0, 25.0, 1.0], [100.0, 75.0, 0.0]]]])
    }

    records = list(
        default_annotate(
            KeypointHead(),
            head_output,
            [image],
            PreprocessingConfig(keep_aspect_ratio=False),
        )
    )

    assert records == [
        {
            "file": str(image),
            "task_name": "",
            "annotation": {
                "instance_id": 0,
                "keypoints": {"keypoints": [(0.25, 0.25, 1), (0.5, 0.75, 0)]},
            },
        }
    ]


def test_classification_uses_each_images_prediction(tmp_path: Path):
    images = [tmp_path / "first.png", tmp_path / "second.png"]
    for image in images:
        _write_image(image)
    head = ClassificationHead(
        dataset_metadata=DatasetMetadata(classes={"": {"cat": 0, "dog": 1}})
    )

    records = list(
        default_annotate(
            head,
            {"classification": torch.tensor([[0.1, 0.9], [0.8, 0.2]])},
            images,
            PreprocessingConfig(),
        )
    )

    assert [_annotation(record)["class"] for record in records] == [
        "dog",
        "cat",
    ]


def test_ocr_annotations_are_emitted_for_each_image(tmp_path: Path):
    images = [tmp_path / "first.png", tmp_path / "second.png"]
    for image in images:
        _write_image(image)

    records = list(
        default_annotate(
            OCRHead(),
            {"ocr": torch.tensor([[[1]], [[2]]])},
            images,
            PreprocessingConfig(),
        )
    )

    assert [_annotation(record)["metadata"]["text"] for record in records] == [
        "1",
        "2",
    ]


def test_annotated_records_skip_out_of_range_boxes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    image = tmp_path / "image.png"
    image.touch()
    record = {
        "file": image,
        "annotation": {"boundingbox": {"x": 3, "y": 0, "w": 1, "h": 1}},
    }
    head = KeypointHead()
    monkeypatch.setattr(head, "annotate", lambda *_: iter([record]))
    debug = Mock()
    monkeypatch.setattr(annotate_utils.logger, "debug", debug)

    assert list(_annotated_records(head, {}, [], PreprocessingConfig())) == []
    debug.assert_called_once()


def test_annotated_records_reraise_other_validation_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    image = tmp_path / "image.png"
    image.touch()
    head = KeypointHead()
    monkeypatch.setattr(
        head,
        "annotate",
        lambda *_: iter([{"file": image, "unexpected": True}]),
    )

    with pytest.raises(ValidationError, match="unexpected"):
        list(_annotated_records(head, {}, [], PreprocessingConfig()))
