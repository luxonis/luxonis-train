import hashlib

import torch
from torch import Tensor

from luxonis_train.attached_modules.visualizers import (
    InstanceSegKeypointVisualizer,
)
from luxonis_train.nodes import BaseNode
from luxonis_train.tasks import Tasks
from luxonis_train.utils.dataset_metadata import DatasetMetadata


class DummyInstanceSegKeypointNode(BaseNode, register=False):
    task = Tasks.INSTANCE_SEGMENTATION_KEYPOINTS

    def forward(self, _: Tensor) -> Tensor: ...


def create_mask(x1: int, y1: int, x2: int, y2: int) -> Tensor:
    mask = torch.zeros(100, 100, dtype=torch.uint8)
    mask[y1:y2, x1:x2] = 1
    return mask


def test_instance_seg_keypoint_visualizer():
    visualizer = InstanceSegKeypointVisualizer(
        node=DummyInstanceSegKeypointNode(
            n_classes=2,
            dataset_metadata=DatasetMetadata(
                classes={"": {"class1": 0, "class2": 1}}
            ),
        )
    )

    canvas = torch.zeros(2, 3, 100, 100, dtype=torch.uint8)

    predictions_bbox = [
        torch.tensor(
            [
                [0, 0, 30, 30, 0.9, 0],
                [30, 30, 70, 70, 0.8, 1],
                [70, 70, 100, 100, 0.7, 0],
            ],
            dtype=torch.float,
        ),
        torch.tensor(
            [
                [0, 0, 30, 30, 0.95, 0],
                [30, 30, 70, 70, 0.85, 1],
                [70, 70, 100, 100, 0.75, 0],
            ],
            dtype=torch.float,
        ),
    ]

    predictions_masks = [
        torch.stack(
            [
                create_mask(0, 0, 30, 30),
                create_mask(30, 30, 70, 70),
                create_mask(70, 70, 100, 100),
            ]
        ),
        torch.stack(
            [
                create_mask(0, 0, 30, 30),
                create_mask(30, 30, 70, 70),
                create_mask(70, 70, 100, 100),
            ]
        ),
    ]

    predictions_keypoints = [
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

    targets_bbox = torch.tensor(
        [
            [0, 0, 0, 0, 30, 30],
            [0, 1, 30, 30, 40, 40],
            [0, 0, 70, 70, 30, 30],
            [1, 0, 0, 0, 30, 30],
            [1, 1, 30, 30, 40, 40],
            [1, 0, 70, 70, 30, 30],
        ],
        dtype=torch.float,
    )
    targets_bbox[:, 2:] /= 100.0

    target_masks = torch.zeros(6, 100, 100, dtype=torch.uint8)
    target_masks[0] = create_mask(0, 0, 30, 30)
    target_masks[1] = create_mask(30, 30, 70, 70)
    target_masks[2] = create_mask(70, 70, 100, 100)
    target_masks[3] = create_mask(0, 0, 30, 30)
    target_masks[4] = create_mask(30, 30, 70, 70)
    target_masks[5] = create_mask(70, 70, 100, 100)

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
        predictions_bbox,
        predictions_masks,
        predictions_keypoints,
        targets_bbox,
        target_masks,
        target_keypoints,
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
        == "ee56331a96bbe5f9b93c8b74f342c34a79691896570c39be8ae8bcd7b371e3ce"
    )
