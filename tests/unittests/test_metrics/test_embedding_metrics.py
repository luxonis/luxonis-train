import pytest
import torch
from torch import Size, Tensor

from luxonis_train.attached_modules.metrics.embedding_metrics import (
    ClosestIsPositiveAccuracy,
    MedianDistances,
)
from luxonis_train.nodes.heads.ghostfacenet_head import GhostFaceNetHead
from luxonis_train.tasks import Tasks


class DummyGhostFaceNetHead(GhostFaceNetHead, register=False):
    task = Tasks.EMBEDDINGS
    original_in_shape: Size = Size([3, 200, 200])
    in_channels: int = 3
    in_width: int = 200

    def forward(self, _: Tensor) -> Tensor: ...


@pytest.mark.parametrize(
    ("embeddings", "labels", "expected"),
    [
        (
            torch.tensor([[0.0, 0.0], [0.0, 0.1], [1.0, 1.0]]),
            torch.tensor([0, 0, 1]),
            1.0,
        ),
        (
            torch.tensor([[0.0, 1.0], [1.0, 1.0], [4.0, 4.0], [3.0, 3.0]]),
            torch.tensor([0, 0, 0, 1]),
            2 / 3,
        ),
        (
            torch.tensor(
                [
                    [0.0, 1.0],
                    [0.0, 2.0],
                    [1.0, 1.0],
                    [4.0, 4.0],
                    [3.0, 3.0],
                    [0.0, 0.0],
                ]
            ),
            torch.tensor([0, 0, 0, 1, 1, 1]),
            5 / 6,
        ),
    ],
)
def test_closest_is_positive_accuracy(
    embeddings: Tensor, labels: Tensor, expected: float
):
    metric = ClosestIsPositiveAccuracy(node=DummyGhostFaceNetHead())
    metric.update(embeddings, labels)
    result = metric.compute()
    assert torch.isclose(result, torch.tensor(expected), atol=1e-6)


@pytest.mark.parametrize(
    ("embeddings", "labels", "expected"),
    [
        (
            torch.tensor(
                [
                    [0.0, 1.0],
                    [0.0, 2.0],
                    [1.0, 1.0],
                    [4.0, 4.0],
                    [3.0, 3.0],
                    [0.0, 0.0],
                ]
            ),
            torch.tensor([0, 0, 0, 1, 1, 1]),
            {
                "MedianDistance": 2.82,
                "MedianClosestDistance": 1.0,
                "MedianClosestPositiveDistance": 1.0,
                "MedianClosestVsClosestPositiveDistance": 0.0,
            },
        ),
        (
            torch.tensor(
                [
                    [0.0, 0.0],
                    [10.0, 10.0],
                    [1.0, 1.0],
                    [2.0, 2.0],
                    [0.0, 0.1],
                ]
            ),
            torch.tensor([0, 0, 1, 1, 1]),
            {
                "MedianDistance": 2.75,
                "MedianClosestDistance": 1.35,
                "MedianClosestPositiveDistance": 1.41,
                "MedianClosestVsClosestPositiveDistance": 1.25,
            },
        ),
        (
            torch.tensor([[0.0, 0.0], [0.0, 3.0], [4.0, 0.0]]),
            torch.tensor([0, 0, 0]),
            {
                "MedianDistance": 4.0,
                "MedianClosestDistance": 3.0,
                "MedianClosestPositiveDistance": 3.0,
                "MedianClosestVsClosestPositiveDistance": 0.0,
            },
        ),
    ],
)
def test_median_distances(
    embeddings: Tensor, labels: Tensor, expected: dict[str, float]
):
    metric = MedianDistances(node=DummyGhostFaceNetHead())
    metric.update(embeddings, labels)
    results = metric.compute()

    for key, value in expected.items():
        assert torch.isclose(results[key], torch.tensor(value), atol=1e-2)
