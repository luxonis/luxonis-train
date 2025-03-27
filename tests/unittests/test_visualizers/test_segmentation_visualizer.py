import hashlib

import torch
from torch import Tensor

from luxonis_train.attached_modules.visualizers import SegmentationVisualizer
from luxonis_train.nodes import BaseNode
from luxonis_train.tasks import Tasks
from luxonis_train.utils.dataset_metadata import DatasetMetadata


class DummySegmentationNode(BaseNode, register=False):
    task = Tasks.SEGMENTATION

    def forward(self, _: Tensor) -> Tensor: ...


def test_segmentation_visualizer():
    visualizer = SegmentationVisualizer(
        node=DummySegmentationNode(
            n_classes=2,
            dataset_metadata=DatasetMetadata(
                classes={"": {"class1": 0, "class2": 1}}
            ),
        ),
        colors=["black", (229, 100, 25)],
    )

    canvas = torch.zeros(2, 3, 10, 10, dtype=torch.uint8)

    predictions = torch.zeros(2, 2, 10, 10, dtype=torch.float)
    targets = torch.zeros(2, 2, 10, 10, dtype=torch.float)

    predictions[0, 1, 2:5, 2:5] = 1.0
    predictions[1, 1, 5:9, 5:9] = 1.0
    targets[0, 1, 1:4, 1:4] = 1.0
    targets[1, 1, 6:9, 6:9] = 1.0

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
        == "87a09bea20b883617b3191ea43365bf4618743feab522f39de063c8f03ff4447"
    )
