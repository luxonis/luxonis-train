from unittest.mock import Mock

import pytest
import torch
from torch import Tensor

from luxonis_train.attached_modules.metrics.ocr_accuracy import OCRAccuracy
from luxonis_train.nodes import OCRCTCHead
from luxonis_train.tasks import Tasks


@pytest.mark.parametrize(
    ("predictions", "targets", "encoded_targets", "expected_ranks"),
    [
        (
            torch.tensor(
                [
                    [
                        [0.8, 0.1, 0.1],  # blank
                        [0.1, 0.7, 0.2],  # a
                        [0.1, 0.2, 0.7],  # b
                        [0.9, 0.05, 0.05],  # blank
                        [0.05, 0.05, 0.9],  # b
                    ]
                ]
            ),
            ["ab"],
            torch.tensor([[1, 2]]),
            {
                "rank_0": torch.tensor(0.0),
                "rank_1": torch.tensor(1.0),
                "rank_2": torch.tensor(0.0),
            },
        ),
        (
            torch.tensor(
                [
                    [
                        [0.03, 0.9, 0.07],  # a
                        [0.7, 0.2, 0.1],  # blank
                        [0.9, 0.05, 0.05],  # blank
                    ],
                    [
                        [0.1, 0.7, 0.2],  # a
                        [0.1, 0.2, 0.7],  # b
                        [0.9, 0.05, 0.05],  # blank
                    ],
                ]
            ),
            ["a", "aab"],
            torch.tensor([[1, 0, 0], [1, 1, 2]]),
            {
                "rank_0": torch.tensor(0.5),
                "rank_1": torch.tensor(0.0),
                "rank_2": torch.tensor(0.5),
            },
        ),
        (
            torch.tensor(
                [
                    [
                        [0.1, 0.3, 0.4, 0.1, 0.1, 0.0],  # h
                        [0.01, 0.9, 0.09, 0.0, 0.0, 0.0],  # e
                        [0.1, 0.2, 0.3, 0.4, 0.0, 0.0],  # l
                        [0.1, 0.2, 0.3, 0.4, 0.0, 0.0],  # l
                        [0.1, 0.2, 0.3, 0.4, 0.0, 0.0],  # l
                        [0.1, 0.2, 0.3, 0.0, 0.4, 0.0],  # o
                    ],
                    [
                        [0.1, 0.3, 0.1, 0.4, 0.1, 0.0],  # l
                        [0.01, 0.9, 0.09, 0.0, 0.0, 0.0],  # e
                        [0.1, 0.2, 0.3, 0.0, 0.0, 0.4],  # w
                        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # blank
                        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # blank
                        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # blank
                    ],
                    [
                        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # blank
                        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # blank
                        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # blank
                        [0.1, 0.3, 0.1, 0.1, 0.4, 0.0],  # o
                        [0.01, 0.0, 0.09, 0.0, 0.0, 0.9],  # w
                        [0.1, 0.2, 0.3, 0.4, 0.0, 0.0],  # l
                    ],
                ]
            ),
            ["hello", "low", "owl"],
            torch.tensor([[2, 1, 3, 3, 4], [3, 4, 5, 0, 0], [4, 5, 3, 0, 0]]),
            {
                "rank_0": torch.tensor(1 / 3),
                "rank_1": torch.tensor(1 / 3),
                "rank_2": torch.tensor(1 / 3),
            },
        ),
    ],
)
def test_ocr_accuracy(
    predictions: Tensor,
    targets: list[str],
    encoded_targets: Tensor,
    expected_ranks: dict[str, Tensor],
):
    mock_node = Mock(spec=OCRCTCHead)
    mock_node.encoder.return_value = encoded_targets
    mock_node.task = Tasks.OCR

    metric = OCRAccuracy(node=mock_node, blank_class=0)

    metric.update(predictions, targets)  # type: ignore

    _, ranks = metric.compute()

    for rank, expected in expected_ranks.items():
        assert torch.isclose(ranks[rank], expected, atol=1e-6)
