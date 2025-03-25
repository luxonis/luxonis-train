import hashlib

import torch
from torch import Tensor

from luxonis_train.attached_modules.visualizers import ClassificationVisualizer
from luxonis_train.nodes import BaseNode
from luxonis_train.tasks import Tasks
from luxonis_train.utils.dataset_metadata import DatasetMetadata


class DummyClassificationNode(BaseNode, register=False):
    task = Tasks.CLASSIFICATION

    def forward(self, _: Tensor) -> Tensor: ...


def test_classification_visualizer():
    visualizer = ClassificationVisualizer(
        node=DummyClassificationNode(
            n_classes=2,
            dataset_metadata=DatasetMetadata(
                classes={"": {"class1": 0, "class2": 1}}
            ),
        )
    )

    canvas = torch.zeros(2, 3, 200, 200, dtype=torch.uint8)

    predictions = torch.tensor([[0.1, 0.9], [0.8, 0.2]], dtype=torch.float)

    targets = torch.tensor([[1], [0]], dtype=torch.long)

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
        == "0c4ec40e35f2a301ccf337237552cf9052792e901a9ecba0a82983f924eb5994"
    )
