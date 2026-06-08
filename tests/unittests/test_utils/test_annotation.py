from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import cv2
import numpy as np
import pytest
import torch
from torch import Tensor

from luxonis_train.utils import annotation as annotation_module
from luxonis_train.utils.annotation import default_annotate


def _head(*labels: str, with_decoder: bool = False) -> Any:
    head = SimpleNamespace(
        name="head",
        task=SimpleNamespace(required_labels=set(labels)),
        task_name="task",
        classes=SimpleNamespace(inverse={0: "zero", 1: "one"}),
    )
    if with_decoder:
        head.decoder = lambda _preds: [("decoded", 0.9)]
    return head


def _preprocessing(keep_aspect_ratio: bool = False) -> Any:
    return SimpleNamespace(
        train_image_size=(8, 8),
        keep_aspect_ratio=keep_aspect_ratio,
    )


def _collect_annotations(
    head: Any,
    head_output: dict[str, Tensor],
    image_paths: list[Path],
    config_preprocessing: Any,
) -> list[dict[str, Any]]:
    return cast(
        list[dict[str, Any]],
        list(
            default_annotate(
                head,
                cast(Any, head_output),
                image_paths,
                config_preprocessing,
            )
        ),
    )


def test_default_annotate_rejects_unknown_label() -> None:
    with pytest.raises(ValueError, match="Unsupported task"):
        _collect_annotations(
            _head("unknown"),
            {},
            [Path("image.jpg")],
            _preprocessing(),
        )


def test_default_annotate_requires_readable_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cv2, "imread", lambda _path: None)

    with pytest.raises(FileNotFoundError, match="Could not read image"):
        _collect_annotations(
            _head("boundingbox"),
            {"boundingbox": torch.empty((1, 0, 6))},
            [Path("missing.jpg")],
            _preprocessing(),
        )


def test_default_annotate_yields_file_for_empty_predictions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cv2, "imread", lambda _path: np.zeros((4, 6, 3), dtype=np.uint8)
    )

    annotations = _collect_annotations(
        _head("boundingbox"),
        {"boundingbox": torch.empty((1, 0, 6))},
        [Path("image.jpg")],
        _preprocessing(),
    )

    assert annotations == [{"file": "image.jpg"}]


def test_default_annotate_bounding_boxes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cv2, "imread", lambda _path: np.zeros((4, 6, 3), dtype=np.uint8)
    )
    monkeypatch.setattr(
        annotation_module,
        "transform_boxes",
        lambda *_args: np.array([[0.1, 0.2, 0.3, 0.4]]),
    )

    annotations = _collect_annotations(
        _head("boundingbox"),
        {"boundingbox": torch.tensor([[[1.0, 2.0, 3.0, 4.0, 0.9, 1.0]]])},
        [Path("image.jpg")],
        _preprocessing(),
    )

    assert annotations == [
        {
            "file": "image.jpg",
            "task_name": "task",
            "annotation": {
                "instance_id": 0,
                "class": "one",
                "boundingbox": {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4},
            },
        }
    ]


def test_default_annotate_keypoints(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        cv2, "imread", lambda _path: np.zeros((4, 6, 3), dtype=np.uint8)
    )
    monkeypatch.setattr(
        annotation_module,
        "transform_keypoints",
        lambda *_args: np.array([[[0.1, 0.2, 0.51], [0.3, 0.4, 1.0]]]),
    )

    annotations = _collect_annotations(
        _head("keypoints"),
        {"keypoints": torch.tensor([[[[1.0, 2.0, 0.5]]]])},
        [Path("image.jpg")],
        _preprocessing(),
    )

    assert annotations == [
        {
            "file": "image.jpg",
            "task_name": "task",
            "annotation": {
                "instance_id": 0,
                "keypoints": {"keypoints": [(0.1, 0.2, 1), (0.3, 0.4, 1)]},
            },
        }
    ]


def test_default_annotate_instance_segmentation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cv2, "imread", lambda _path: np.zeros((4, 6, 3), dtype=np.uint8)
    )
    monkeypatch.setattr(
        annotation_module,
        "transform_masks",
        lambda *_args: np.array([[[1, 0], [0, 1]]], dtype=np.uint8),
    )

    annotations = _collect_annotations(
        _head("instance_segmentation"),
        {"instance_segmentation": torch.ones((1, 1, 2, 2))},
        [Path("image.jpg")],
        _preprocessing(),
    )

    mask = annotations[0]["annotation"]["instance_segmentation"]["mask"]
    assert annotations[0]["file"] == "image.jpg"
    assert annotations[0]["annotation"]["instance_id"] == 0
    assert mask.dtype == np.bool_
    assert mask.tolist() == [[True, False], [False, True]]


def test_default_annotate_segmentation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cv2, "imread", lambda _path: np.zeros((4, 6, 3), dtype=np.uint8)
    )
    monkeypatch.setattr(
        annotation_module,
        "seg_output_to_bool",
        lambda _preds: torch.tensor([[[True, False], [False, True]]]),
    )
    monkeypatch.setattr(
        annotation_module,
        "transform_masks",
        lambda *_args: np.array([[[1, 0], [0, 1]]], dtype=np.uint8),
    )

    annotations = _collect_annotations(
        _head("segmentation"),
        {"segmentation": torch.ones((1, 1, 2, 2))},
        [Path("image.jpg")],
        _preprocessing(),
    )

    assert annotations[0]["annotation"]["class"] == "zero"
    assert annotations[0]["annotation"]["segmentation"]["mask"].tolist() == [
        [True, False],
        [False, True],
    ]


def test_default_annotate_classification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cv2, "imread", lambda _path: np.zeros((4, 6, 3), dtype=np.uint8)
    )

    annotations = _collect_annotations(
        _head("classification"),
        {"classification": torch.tensor([[[0.1, 0.9]]])},
        [Path("image.jpg")],
        _preprocessing(),
    )

    assert annotations == [
        {
            "file": "image.jpg",
            "task_name": "task",
            "annotation": {"class": "one"},
        }
    ]


def test_default_annotate_text_requires_decoder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cv2, "imread", lambda _path: np.zeros((4, 6, 3), dtype=np.uint8)
    )

    with pytest.raises(ValueError, match="decoder"):
        _collect_annotations(
            _head("boundingbox", "text"),
            {
                "boundingbox": torch.tensor(
                    [[[1.0, 2.0, 3.0, 4.0, 0.9, 1.0]]]
                ),
                "ocr": torch.zeros((1, 2, 3)),
            },
            [Path("image.jpg")],
            _preprocessing(),
        )


def test_default_annotate_text(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        cv2, "imread", lambda _path: np.zeros((4, 6, 3), dtype=np.uint8)
    )
    monkeypatch.setattr(
        annotation_module,
        "transform_boxes",
        lambda *_args: np.array([[0.1, 0.2, 0.3, 0.4]]),
    )

    annotations = _collect_annotations(
        _head("boundingbox", "text", with_decoder=True),
        {
            "boundingbox": torch.tensor([[[1.0, 2.0, 3.0, 4.0, 0.9, 1.0]]]),
            "ocr": torch.zeros((1, 2, 3)),
        },
        [Path("image.jpg")],
        _preprocessing(),
    )

    assert {
        "file": "image.jpg",
        "task_name": "task",
        "annotation": {"metadata": {"text": "decoded"}},
    } in annotations
