import hashlib

import torch
from torch import Tensor

from luxonis_train.attached_modules.visualizers import FOMOVisualizer
from luxonis_train.nodes import BaseNode
from luxonis_train.tasks import Tasks
from luxonis_train.utils.dataset_metadata import DatasetMetadata


class DummyFOMONode(BaseNode, register=False):
    task = Tasks.FOMO

    def forward(self, _: Tensor) -> Tensor: ...


def test_fomo_visualizer():
    visualizer = FOMOVisualizer(
        node=DummyFOMONode(
            n_classes=2,
            dataset_metadata=DatasetMetadata(
                classes={"": {"class1": 0, "class2": 1}}
            ),
        )
    )

    canvas = torch.zeros(2, 3, 100, 100, dtype=torch.uint8)

    predicted_keypoints = [
        torch.tensor(
            [
                [[15, 15, 1]],
                [[40, 40, 1]],
                [[75, 75, 1]],
            ],
            dtype=torch.float,
        ),
        torch.tensor(
            [
                [[15, 15, 1]],
                [[40, 40, 1]],
                [[75, 75, 1]],
            ],
            dtype=torch.float,
        ),
    ]

    target_boundingboxes = torch.tensor(
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
    target_boundingboxes[:, 2:] /= 100.0

    targets_viz, predictions_viz = visualizer(
        canvas.clone(),
        canvas.clone(),
        predicted_keypoints,
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
        == "1b5954436143c02eae3fa3fd4400a0785d95a4ce1f9683a948d44abd027dce66"
    )
