import hashlib

import torch
from torch import Tensor

from luxonis_train.attached_modules.visualizers import (
    InstanceSegmentationVisualizer,
)
from luxonis_train.nodes import BaseNode
from luxonis_train.tasks import Tasks
from luxonis_train.utils.dataset_metadata import DatasetMetadata


class DummyInstanceSegmentationNode(BaseNode, register=False):
    task = Tasks.INSTANCE_SEGMENTATION

    def forward(self, _: Tensor) -> Tensor: ...


def create_mask(x1: int, y1: int, x2: int, y2: int) -> Tensor:
    mask = torch.zeros(100, 100, dtype=torch.uint8)
    mask[y1:y2, x1:x2] = 1
    return mask


def test_instance_segmentation_visualizer():
    visualizer = InstanceSegmentationVisualizer(
        node=DummyInstanceSegmentationNode(
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

    targets_viz, predictions_viz = visualizer(
        canvas.clone(),
        canvas.clone(),
        predictions_bbox,
        predictions_masks,
        targets_bbox,
        target_masks,
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
        == "fa4ebac026d21765e6687db04f8f2b9a7947a53fd22091d2a3d47d0f6ec1eb6a"
    )
