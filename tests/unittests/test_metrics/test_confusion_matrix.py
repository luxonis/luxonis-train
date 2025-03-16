import torch
from torch import Tensor

from luxonis_train.attached_modules.metrics.confusion_matrix import (
    DetectionConfusionMatrix,
)
from luxonis_train.nodes import BaseNode
from luxonis_train.tasks import Tasks


def test_compute_detection_confusion_matrix_specific_case():
    class DummyNodeDetection(BaseNode):
        task = Tasks.BOUNDINGBOX

        def forward(self, _: Tensor) -> Tensor: ...

    metric = DetectionConfusionMatrix(node=DummyNodeDetection(n_classes=3))

    preds = [torch.empty((0, 6)) for _ in range(3)]
    preds.append(
        torch.tensor(
            [
                [10, 20, 30, 50, 0.8, 2],
                [10, 21, 30, 50, 0.8, 1],
                [10, 20, 30, 50, 0.8, 1],
                [51, 61, 71, 78, 0.9, 2],
            ]
        )
    )

    # Targets: ground truth for 4 images
    targets = torch.tensor(
        [
            [3, 1, 10, 20, 30, 50],
            [0, 1, 10, 20, 30, 40],
            [1, 2, 50, 60, 70, 80],
            [2, 2, 10, 60, 70, 80],
            [3, 2, 50, 60, 70, 80],
        ]
    )

    expected_cm = [
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 1, 1, 0],
        [0, 1, 3, 1],
    ]
    metric._update(preds, targets)
    metric._update([torch.empty(0, 6)], torch.empty((0, 6)))

    computed_cm = metric.compute().tolist()

    assert computed_cm == expected_cm
