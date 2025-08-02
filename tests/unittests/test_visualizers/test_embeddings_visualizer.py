import hashlib

import torch
from torch import Tensor

from luxonis_train.attached_modules.visualizers import EmbeddingsVisualizer
from luxonis_train.nodes import BaseNode
from luxonis_train.tasks import Tasks


class DummyEmbeddingNode(BaseNode, register=False):
    task = Tasks.EMBEDDINGS

    def forward(self, x: Tensor) -> Tensor:  # pragma: no cover
        return x


def test_embeddings_visualizer():
    visualizer = EmbeddingsVisualizer(node=DummyEmbeddingNode())

    canvas = torch.zeros(1, 3, 100, 100, dtype=torch.uint8)

    predictions = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 3.0, 4.0, 5.0],
            [3.0, 4.0, 5.0, 6.0],
            [4.0, 5.0, 6.0, 7.0],
            [5.0, 6.0, 7.0, 8.0],
            [6.0, 7.0, 8.0, 9.0],
            [7.0, 8.0, 9.0, 10.0],
            [8.0, 9.0, 10.0, 11.0],
            [9.0, 10.0, 11.0, 12.0],
            [10.0, 11.0, 12.0, 13.0],
        ],
        dtype=torch.float,
    )

    target = torch.tensor([0, 0, 1, 1, 2, 2, 0, 1, 2, 1], dtype=torch.int)

    kdeplot, scatterplot = visualizer(
        canvas.clone(), canvas.clone(), predictions, target
    )

    combined_viz = (
        torch.cat([kdeplot, scatterplot], dim=0).cpu().numpy().tobytes()
    )
    computed_hash = hashlib.sha256(combined_viz).hexdigest()

    expected_hash = (
        "3f2ac86f1c7463ca7e75ba41b7fc28189da1f073d27599bb71258a8645bbbaf9"
    )
    assert computed_hash == expected_hash
