import hashlib

import torch
from torch import Tensor

from luxonis_train.attached_modules.visualizers import KeypointVisualizer
from luxonis_train.nodes import BaseNode
from luxonis_train.tasks import Tasks
from luxonis_train.utils.dataset_metadata import DatasetMetadata


class DummyKeypointsNode(BaseNode, register=False):
    task = Tasks.INSTANCE_KEYPOINTS

    def forward(self, _: Tensor) -> Tensor: ...


def test_keypoint_visualizer():
    visualizer = KeypointVisualizer(
        node=DummyKeypointsNode(
            n_classes=2,
            dataset_metadata=DatasetMetadata(
                classes={"": {"class1": 0, "class2": 1}}
            ),
        )
    )

    canvas = torch.zeros(2, 3, 100, 100, dtype=torch.uint8)

    boundingboxes = [
        torch.tensor(
            [
                [0, 0, 30, 30, 0.9, 0],
                [30, 30, 60, 60, 0.8, 1],
                [70, 70, 100, 100, 0.7, 0],
            ],
            dtype=torch.float,
        ),
        torch.tensor(
            [
                [0, 0, 30, 30, 0.9, 0],
                [30, 30, 60, 60, 0.8, 1],
                [70, 70, 100, 100, 0.7, 0],
            ],
            dtype=torch.float,
        ),
    ]

    target_boundingboxes = torch.tensor(
        [
            [0, 0, 0, 0, 30, 30],
            [0, 1, 30, 30, 30, 30],
            [0, 0, 70, 70, 30, 30],
            [1, 0, 0, 0, 30, 30],
            [1, 1, 30, 30, 30, 30],
            [1, 0, 70, 70, 30, 30],
        ],
        dtype=torch.float,
    )
    target_boundingboxes[:, 2:] /= 100.0

    keypoints = [
        torch.tensor(
            [
                [[10, 10, 1], [20, 10, 1], [10, 20, 1], [20, 20, 1]],
                [[35, 35, 1], [45, 35, 1], [35, 45, 1], [45, 45, 1]],
                [[70, 70, 1], [80, 70, 1], [70, 80, 1], [80, 80, 1]],
            ],
            dtype=torch.float,
        ),
        torch.tensor(
            [
                [[10, 10, 1], [20, 10, 1], [10, 20, 1], [20, 20, 1]],
                [[35, 35, 1], [45, 35, 1], [35, 45, 1], [45, 45, 1]],
                [[70, 70, 1], [80, 70, 1], [70, 80, 1], [80, 80, 1]],
            ],
            dtype=torch.float,
        ),
    ]

    target_keypoints = torch.tensor(
        [
            [0, 0.10, 0.10, 1, 0.20, 0.10, 1, 0.10, 0.20, 1, 0.20, 0.20, 1],
            [0, 0.35, 0.35, 1, 0.45, 0.35, 1, 0.35, 0.45, 1, 0.45, 0.45, 1],
            [0, 0.70, 0.70, 1, 0.80, 0.70, 1, 0.70, 0.80, 1, 0.80, 0.80, 1],
            [1, 0.10, 0.10, 1, 0.20, 0.10, 1, 0.10, 0.20, 1, 0.20, 0.20, 1],
            [1, 0.35, 0.35, 1, 0.45, 0.35, 1, 0.35, 0.45, 1, 0.45, 0.45, 1],
            [1, 0.70, 0.70, 1, 0.80, 0.70, 1, 0.70, 0.80, 1, 0.80, 0.80, 1],
        ],
        dtype=torch.float,
    )

    targets_viz, predictions_viz = visualizer(
        canvas.clone(),
        canvas.clone(),
        keypoints,
        boundingboxes,
        target_keypoints,
        target_boundingboxes,
    )

    combined_viz = (
        torch.cat([targets_viz, predictions_viz], dim=0)
        .cpu()
        .numpy()
        .tobytes()
    )
    computed_hash = hashlib.sha256(combined_viz).hexdigest()

    assert (
        computed_hash
        == "07f0dcac054481d8f11ac04a5bcab1544e0f6893f5fe7d1634d6e1f98cd65bf3"
    )
