import pytest
import torch
from torch import Tensor

from luxonis_train.attached_modules.metrics.object_keypoint_similarity import (
    ObjectKeypointSimilarity,
)
from luxonis_train.nodes import BaseNode
from luxonis_train.tasks import Tasks
from luxonis_train.utils.dataset_metadata import DatasetMetadata

from .test_utils import convert_bboxes_to_xyxy_and_normalize, normalize_kpts


@pytest.mark.parametrize(
    (
        "keypoints",
        "target_keypoints",
        "target_boundingbox",
        "expected",
    ),
    [
        (
            [
                torch.tensor(
                    [
                        [
                            [11, 11, 1],
                            [21, 26, 1],
                            [24, 48, 1],
                            [29, 35, 1],
                        ],
                        [
                            [14, 21, 1],
                            [22, 33, 1],
                            [33, 46, 1],
                            [42, 39, 1],
                        ],
                        [
                            [12, 21, 1],
                            [17, 31, 1],
                            [24, 36, 1],
                            [16, 27, 1],
                        ],
                    ]
                ),
            ],
            torch.tensor(
                [
                    [0, 12, 12, 1, 20, 25, 1, 25, 50, 1, 30, 37, 1],
                    [0, 15, 20, 1, 25, 35, 1, 35, 45, 1, 40, 40, 1],
                    [0, 10, 22, 1, 18, 30, 1, 25, 35, 1, 16, 28, 1],
                ]
            ),
            torch.tensor(
                [
                    [0, 0, 10, 10, 50, 70],
                    [0, 0, 15, 15, 65, 65],
                    [0, 0, 5, 20, 35, 40],
                ]
            ),
            torch.tensor(0.70),
        ),
        (
            [
                torch.tensor(
                    [
                        [
                            [42, 63, 0.8],
                            [87, 59, 0.85],
                            [66, 86, 0.9],
                            [64, 108, 0.75],
                        ],
                    ]
                ),
            ],
            torch.tensor(
                [
                    [0, 40, 60, 2, 90, 60, 2, 65, 85, 2, 65, 110, 2],
                ]
            ),
            torch.tensor(
                [
                    [0, 0, 30, 40, 30 + 80, 40 + 90],
                ]
            ),
            torch.tensor(0.86),
        ),
    ],
)
def test_compute_object_keypoint_similarity(
    keypoints: list[Tensor],
    target_keypoints: Tensor,
    target_boundingbox: Tensor,
    expected: Tensor,
):
    class DummyNodeKeypoints(BaseNode, register=False):
        task = Tasks.INSTANCE_KEYPOINTS

        def forward(self, _: Tensor) -> Tensor: ...

    image_size = torch.Size([3, 200, 200])
    sigmas = [0.04, 0.04, 0.04, 0.04]
    area_factor = 0.53
    metric = ObjectKeypointSimilarity(
        sigmas=sigmas,
        area_factor=area_factor,
        node=DummyNodeKeypoints(
            n_classes=2,
            n_keypoints=4,
            dataset_metadata=DatasetMetadata(
                classes={"": {"class1": 0, "class2": 1}}
            ),
            original_in_shape=image_size,
        ),
    )

    target_boundingbox = convert_bboxes_to_xyxy_and_normalize(
        target_boundingbox, image_size
    )
    target_keypoints = normalize_kpts(target_keypoints, image_size)

    metric.update(keypoints, target_boundingbox, target_keypoints)
    result = metric.compute()
    assert torch.isclose(result, expected, atol=5e-3)
