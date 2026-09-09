from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from torch import Tensor

from luxonis_train.config.config import PreprocessingConfig
from luxonis_train.nodes import BaseHead
from luxonis_train.tasks import Tasks
from luxonis_train.typing import Packet
from luxonis_train.utils.annotation import default_annotate


class KeypointHead(BaseHead, register=False):
    task = Tasks.KEYPOINTS

    def forward(self, _: Tensor) -> Tensor: ...


class EmbeddingHead(BaseHead, register=False):
    task = Tasks.EMBEDDINGS

    def forward(self, _: Tensor) -> Tensor: ...


def test_unsupported_task_is_rejected():
    with pytest.raises(ValueError, match="Unsupported task"):
        list(default_annotate(EmbeddingHead(), {}, [], PreprocessingConfig()))


def test_keypoints_are_normalized_to_the_original_image(tmp_path: Path):
    image = tmp_path / "image.png"
    cv2.imwrite(str(image), np.zeros((100, 200, 3), dtype=np.uint8))
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
