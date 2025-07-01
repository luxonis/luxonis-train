import hashlib

import torch
from torch import Tensor

from luxonis_train.attached_modules.visualizers import BBoxVisualizer
from luxonis_train.nodes import BaseNode
from luxonis_train.tasks import Tasks
from luxonis_train.utils.dataset_metadata import DatasetMetadata


class DummyBBoxNode(BaseNode, register=False):
    task = Tasks.BOUNDINGBOX

    def forward(self, _: Tensor) -> Tensor: ...


def test_bbox_visualizer():
    visualizer = BBoxVisualizer(
        node=DummyBBoxNode(
            n_classes=2,
            dataset_metadata=DatasetMetadata(
                classes={"": {"class1": 0, "class2": 1}}
            ),
        )
    )

    canvas = torch.zeros(2, 3, 100, 100, dtype=torch.uint8)
    targets = torch.tensor(
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
    targets[:, 2:] /= 100.0

    predictions = [
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
                [0, 0, 30, 30, 0.9, 0],
                [30, 30, 70, 70, 0.8, 1],
                [70, 70, 100, 100, 0.7, 0],
            ],
            dtype=torch.float,
        ),
    ]

    targets_viz, predictions_viz = visualizer(
        canvas.clone(), canvas.clone(), predictions, targets
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
        == "5fbd68cb6486681905d515497ac7b08ce3f5425caa2614daf0a0932c9d72fac1"
    )
