import pytest
import torch
from torch import Tensor

from luxonis_train.attached_modules.metrics.confusion_matrix import (
    DetectionConfusionMatrix,
)
from luxonis_train.nodes import BaseNode
from luxonis_train.tasks import Tasks


@pytest.mark.parametrize(
    ("predictions", "targets", "expected"),
    [
        (
            [torch.empty((0, 6))],
            torch.empty((0, 6)),
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 1],
            ],
        ),
        (
            [torch.tensor([[70, 60, 110, 80, 0.8, 2]])],
            torch.empty((0, 6)),
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 0, 0],
            ],
        ),
        (
            [torch.empty((0, 6))],
            torch.tensor([[0, 2, 50, 60, 70, 80]]),
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 1, 0],
            ],
        ),
        (
            [torch.tensor([[70, 60, 110, 80, 0.8, 2]])],
            torch.tensor([[0, 2, 50, 60, 70, 80]]),
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0],
            ],
        ),
        (
            [torch.tensor([[50, 60, 70, 80, 0.8, 1]])],
            torch.tensor([[0, 2, 50, 60, 70, 90]]),
            [
                [0, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
        ),
        (
            [torch.tensor([[50, 60, 70, 80, 0.8, 1]])],
            torch.tensor(
                [
                    [0, 2, 50, 60, 70, 90],
                    [0, 2, 50, 60, 70, 105],
                ]
            ),
            [
                [0, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 0],
                [0, 0, 1, 0],
            ],
        ),
        (
            [
                torch.empty((0, 6)),
                torch.empty((0, 6)),
                torch.empty((0, 6)),
                torch.tensor(
                    [
                        [10, 20, 30, 50, 0.8, 2],
                        [10, 21, 30, 50, 0.8, 1],
                        [10, 20, 30, 50, 0.8, 1],
                        [51, 61, 71, 78, 0.9, 2],
                    ]
                ),
            ],
            torch.tensor(
                [
                    [0, 1, 10, 20, 30, 40],
                    [1, 2, 50, 60, 70, 80],
                    [2, 2, 10, 60, 70, 80],
                    [3, 1, 10, 20, 30, 50],
                    [3, 2, 50, 60, 70, 80],
                ]
            ),
            [
                [0, 0, 0, 0],
                [0, 0, 0, 2],
                [0, 1, 1, 0],
                [0, 1, 2, 0],
            ],
        ),
        (
            [
                torch.tensor([[10, 20, 30, 40, 1.0, 1]]),
                torch.tensor([[50, 60, 70, 80, 1.0, 2]]),
                torch.tensor([[10, 60, 70, 80, 1.0, 0]]),
                torch.tensor(
                    [
                        [10, 20, 30, 50, 0.8, 1],
                        [51, 61, 71, 78, 0.9, 2],
                    ]
                ),
            ],
            torch.tensor(
                [
                    [0, 1, 10, 20, 30, 40],
                    [1, 2, 50, 60, 70, 80],
                    [2, 0, 10, 60, 70, 80],
                    [3, 1, 10, 20, 30, 50],
                    [3, 2, 50, 60, 70, 80],
                ]
            ),
            [
                [1, 0, 0, 0],
                [0, 2, 0, 0],
                [0, 0, 2, 0],
                [0, 0, 0, 0],
            ],
        ),
        (
            [
                torch.tensor([[10, 20, 30, 40, 1.0, 0]]),
                torch.tensor([[50, 60, 70, 80, 1.0, 0]]),
                torch.tensor([[10, 60, 70, 80, 1.0, 0]]),
                torch.tensor(
                    [
                        [10, 20, 30, 50, 0.8, 0],
                        [51, 61, 71, 78, 0.9, 0],
                    ]
                ),
            ],
            torch.tensor(
                [
                    [0, 1, 10, 20, 30, 40],
                    [1, 2, 50, 60, 70, 80],
                    [2, 0, 10, 60, 70, 80],
                    [3, 1, 10, 20, 30, 50],
                    [3, 2, 50, 60, 70, 80],
                ]
            ),
            [
                [1, 2, 2, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
        ),
    ],
)
def test_compute_detection_confusion_matrix_specific_case(
    predictions: list[Tensor], targets: Tensor, expected: list[list[int]]
):
    class DummyNodeDetection(BaseNode, register=False):
        task = Tasks.BOUNDINGBOX

        def forward(self, _: Tensor) -> Tensor: ...

    metric = DetectionConfusionMatrix(node=DummyNodeDetection(n_classes=3))

    metric._update(predictions, targets)
    assert metric.compute()["confusion_matrix"].tolist() == expected
