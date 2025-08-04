from typing import Literal

import pytest
import torch
from torch import Tensor

from luxonis_train.attached_modules.metrics.dice_coefficient import (
    DiceCoefficient,
)
from luxonis_train.attached_modules.metrics.mean_iou import MIoU
from luxonis_train.nodes import BaseNode
from luxonis_train.tasks import Tasks


class DummyNodeSegmentation(BaseNode, register=False):
    task = Tasks.SEGMENTATION

    def forward(self, _: Tensor) -> Tensor: ...


@pytest.mark.parametrize(
    ("predictions", "targets", "expected"),
    [
        (
            torch.tensor(
                [
                    [[1.0, 0.0], [0.0, 1.0]],
                    [[0.0, 1.0], [1.0, 0.0]],
                ]
            ).unsqueeze(0),
            torch.tensor(
                [
                    [[1.0, 0.0], [0.0, 1.0]],
                    [[0.0, 1.0], [1.0, 0.0]],
                ]
            ).unsqueeze(0),
            torch.tensor(1.0),
        ),
        (
            torch.tensor(
                [
                    [[1.0, 0.0], [0.0, 0.0]],
                    [[0.0, 1.0], [1.0, 1.0]],
                ]
            ).unsqueeze(0),
            torch.tensor(
                [
                    [[1.0, 0.0], [0.0, 1.0]],
                    [[0.0, 1.0], [1.0, 0.0]],
                ]
            ).unsqueeze(0),
            torch.tensor(0.75),
        ),
        (
            torch.tensor(
                [
                    [[1.0, 1.0], [1.0, 1.0]],
                    [[0.0, 0.0], [0.0, 0.0]],
                ]
            ).unsqueeze(0),
            torch.tensor(
                [
                    [[0.0, 0.0], [0.0, 0.0]],
                    [[1.0, 1.0], [1.0, 1.0]],
                ]
            ).unsqueeze(0),
            torch.tensor(0.0),
        ),
        (
            torch.tensor(
                [
                    [[0.8, 0.2], [0.3, 0.4]],
                    [[0.2, 0.8], [0.7, 0.6]],
                ]
            ).unsqueeze(0),
            torch.tensor(
                [
                    [[1.0, 0.0], [0.0, 0.0]],
                    [[0.0, 1.0], [1.0, 1.0]],
                ]
            ).unsqueeze(0),
            torch.tensor(1.0),
        ),
        (
            torch.tensor(
                [
                    [[0.0, 0.0], [0.0, 0.0]],
                    [[0.0, 0.0], [0.0, 0.0]],
                ]
            ).unsqueeze(0),
            torch.tensor(
                [
                    [[0.0, 0.0], [0.0, 0.0]],
                    [[0.0, 0.0], [0.0, 0.0]],
                ]
            ).unsqueeze(0),
            torch.tensor(0.0),
        ),
    ],
)
def test_dice_coefficient_one_hot(
    predictions: Tensor, targets: Tensor, expected: Tensor
):
    num_classes = predictions.shape[1]

    metric = DiceCoefficient(
        num_classes=num_classes,
        include_background=True,
        average="micro",
        input_format="one-hot",
        node=DummyNodeSegmentation(n_classes=num_classes),
    )

    metric.update(predictions, targets)
    result = metric.compute()
    assert torch.isclose(result, expected, atol=1e-4), (
        f"Expected {expected}, got {result}"
    )


@pytest.mark.parametrize(
    ("predictions", "targets", "expected"),
    [
        (
            torch.tensor(
                [
                    [0, 1, 1, 0],
                    [1, 0, 0, 1],
                ]
            ).unsqueeze(0),
            torch.tensor(
                [
                    [0, 1, 1, 0],
                    [1, 0, 0, 1],
                ]
            ).unsqueeze(0),
            torch.tensor(1.0),
        ),
        (
            torch.tensor(
                [
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                ]
            ).unsqueeze(0),
            torch.tensor(
                [
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ]
            ).unsqueeze(0),
            torch.tensor(1.0),
        ),
        (
            torch.tensor(
                [
                    [0, 0, 1, 1],
                    [1, 1, 0, 0],
                ]
            ).unsqueeze(0),
            torch.tensor(
                [
                    [0, 1, 1, 0],
                    [1, 0, 0, 1],
                ]
            ).unsqueeze(0),
            torch.tensor(0.5),
        ),
    ],
)
def test_dice_coefficient_index(
    predictions: Tensor, targets: Tensor, expected: Tensor
):
    metric = DiceCoefficient(
        num_classes=2,
        include_background=True,
        average="micro",
        input_format="index",
        node=DummyNodeSegmentation(n_classes=2),
    )

    metric.update(predictions, targets)
    result = metric.compute()
    assert torch.isclose(result, expected, atol=1e-4), (
        f"Expected {expected}, got {result}"
    )


@pytest.mark.parametrize(
    ("predictions", "targets", "average", "expected"),
    [
        (
            torch.tensor(
                [
                    [[1.0, 0.0], [0.0, 0.0]],
                    [[0.0, 1.0], [1.0, 1.0]],
                ]
            ).unsqueeze(0),
            torch.tensor(
                [
                    [[1.0, 0.0], [0.0, 1.0]],
                    [[0.0, 1.0], [1.0, 0.0]],
                ]
            ).unsqueeze(0),
            "micro",
            torch.tensor(0.75),
        ),
        (
            torch.tensor(
                [
                    [[1.0, 0.0], [1.0, 0.0]],
                    [[0.0, 1.0], [0.0, 1.0]],
                ]
            ).unsqueeze(0),
            torch.tensor(
                [
                    [[1.0, 0.0], [0.0, 1.0]],
                    [[0.0, 1.0], [1.0, 0.0]],
                ]
            ).unsqueeze(0),
            "macro",
            torch.tensor(0.5),
        ),
        (
            torch.tensor(
                [
                    [[1.0, 0.0], [0.0, 0.0]],
                    [[0.0, 1.0], [1.0, 1.0]],
                ]
            ).unsqueeze(0),
            torch.tensor(
                [
                    [[1.0, 0.0], [0.0, 1.0]],
                    [[0.0, 1.0], [1.0, 0.0]],
                ]
            ).unsqueeze(0),
            None,
            torch.tensor([2 / 3, 0.8000]),
        ),
    ],
)
def test_dice_coefficient_averaging(
    predictions: Tensor,
    targets: Tensor,
    average: Literal["micro", "macro", "weighted", "none"],
    expected: Tensor,
):
    metric = DiceCoefficient(
        num_classes=2,
        include_background=True,
        average=average,
        input_format="one-hot",
        node=DummyNodeSegmentation(n_classes=2),
    )

    metric.update(predictions, targets)
    result = metric.compute()

    if average is None:
        assert torch.allclose(result, expected, atol=1e-4), (
            f"Expected {expected}, got {result}"
        )
    else:
        assert torch.isclose(result, expected, atol=1e-4), (
            f"Expected {expected}, got {result}"
        )


@pytest.mark.parametrize(
    ("predictions", "targets", "include_background", "expected"),
    [
        (
            torch.tensor(
                [
                    [[1.0, 0.0], [0.0, 0.0]],
                    [[0.0, 1.0], [1.0, 1.0]],
                ]
            ).unsqueeze(0),
            torch.tensor(
                [
                    [[1.0, 0.0], [0.0, 1.0]],
                    [[0.0, 1.0], [1.0, 0.0]],
                ]
            ).unsqueeze(0),
            True,
            torch.tensor(0.75),
        ),
        (
            torch.tensor(
                [
                    [[1.0, 0.0], [0.0, 0.0]],
                    [[0.0, 1.0], [1.0, 1.0]],
                ]
            ).unsqueeze(0),
            torch.tensor(
                [
                    [[1.0, 0.0], [0.0, 1.0]],
                    [[0.0, 1.0], [1.0, 0.0]],
                ]
            ).unsqueeze(0),
            False,
            torch.tensor(0.8),
        ),
    ],
)
def test_dice_coefficient_background(
    predictions: Tensor,
    targets: Tensor,
    include_background: bool,
    expected: Tensor,
):
    metric = DiceCoefficient(
        num_classes=2,
        include_background=include_background,
        average="micro",
        input_format="one-hot",
        node=DummyNodeSegmentation(n_classes=2),
    )

    metric.update(predictions, targets)
    result = metric.compute()
    assert torch.isclose(result, expected, atol=1e-4), (
        f"Expected {expected}, got {result}"
    )


@pytest.mark.parametrize(
    ("predictions", "targets", "expected"),
    [
        (
            torch.tensor(
                [
                    [[1.0, 0.0], [0.0, 1.0]],
                    [[0.0, 1.0], [1.0, 0.0]],
                ]
            ).unsqueeze(0),
            torch.tensor(
                [
                    [[1.0, 0.0], [0.0, 1.0]],
                    [[0.0, 1.0], [1.0, 0.0]],
                ]
            ).unsqueeze(0),
            torch.tensor(1.0),
        ),
        (
            torch.tensor(
                [
                    [[1.0, 1.0], [1.0, 1.0]],
                    [[0.0, 0.0], [0.0, 0.0]],
                ]
            ).unsqueeze(0),
            torch.tensor(
                [
                    [[0.0, 0.0], [0.0, 0.0]],
                    [[1.0, 1.0], [1.0, 1.0]],
                ]
            ).unsqueeze(0),
            torch.tensor(0.0),
        ),
        (
            torch.tensor(
                [
                    [[0.8, 0.2], [0.3, 0.4]],
                    [[0.2, 0.8], [0.7, 0.6]],
                ]
            ).unsqueeze(0),
            torch.tensor(
                [
                    [[1.0, 0.0], [0.0, 0.0]],
                    [[0.0, 1.0], [1.0, 1.0]],
                ]
            ).unsqueeze(0),
            torch.tensor(1.0),
        ),
    ],
)
def test_mean_iou_one_hot(
    predictions: Tensor, targets: Tensor, expected: Tensor
):
    num_classes = predictions.shape[1]

    metric = MIoU(
        num_classes=num_classes,
        include_background=True,
        input_format="one-hot",
        node=DummyNodeSegmentation(n_classes=num_classes),
    )

    metric.update(predictions, targets)
    result = metric.compute()
    assert torch.isclose(result, expected, atol=1e-4), (
        f"Expected {expected}, got {result}"
    )


@pytest.mark.parametrize(
    ("predictions", "targets", "expected"),
    [
        (
            torch.tensor(
                [
                    [0, 1, 1, 0],
                    [1, 0, 0, 1],
                ]
            ).unsqueeze(0),
            torch.tensor(
                [
                    [0, 1, 1, 0],
                    [1, 0, 0, 1],
                ]
            ).unsqueeze(0),
            torch.tensor(1.0),
        ),
        (
            torch.tensor(
                [
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                ]
            ).unsqueeze(0),
            torch.tensor(
                [
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ]
            ).unsqueeze(0),
            torch.tensor(0.5),
        ),
        (
            torch.tensor(
                [
                    [0, 0, 1, 1],
                    [1, 1, 0, 0],
                ]
            ).unsqueeze(0),
            torch.tensor(
                [
                    [0, 1, 1, 0],
                    [1, 0, 0, 1],
                ]
            ).unsqueeze(0),
            torch.tensor(1 / 3),
        ),
    ],
)
def test_mean_iou_index(
    predictions: Tensor, targets: Tensor, expected: Tensor
):
    metric = MIoU(
        num_classes=2,
        include_background=True,
        input_format="index",
        node=DummyNodeSegmentation(n_classes=2),
    )

    metric.update(predictions, targets)
    result = metric.compute()
    assert torch.isclose(result, expected, atol=1e-4), (
        f"Expected {expected}, got {result}"
    )


@pytest.mark.parametrize(
    ("predictions", "targets", "include_background", "expected"),
    [
        (
            torch.tensor(
                [
                    [[1.0, 0.0], [0.0, 0.0]],
                    [[0.0, 1.0], [1.0, 1.0]],
                ]
            ).unsqueeze(0),
            torch.tensor(
                [
                    [[1.0, 1.0], [0.0, 1.0]],
                    [[0.0, 1.0], [1.0, 0.0]],
                ]
            ).unsqueeze(0),
            True,
            torch.tensor(0.5),
        ),
        (
            torch.tensor(
                [
                    [[1.0, 0.0], [0.0, 0.0]],
                    [[0.0, 1.0], [1.0, 1.0]],
                ]
            ).unsqueeze(0),
            torch.tensor(
                [
                    [[0.0, 0.0], [0.0, 1.0]],
                    [[0.0, 1.0], [1.0, 0.0]],
                ]
            ).unsqueeze(0),
            False,
            torch.tensor(2 / 3),
        ),
    ],
)
def test_mean_iou_background(
    predictions: Tensor,
    targets: Tensor,
    include_background: bool,
    expected: Tensor,
):
    metric = MIoU(
        num_classes=2,
        include_background=include_background,
        input_format="one-hot",
        node=DummyNodeSegmentation(n_classes=2),
    )

    metric.update(predictions, targets)
    result = metric.compute()
    assert torch.isclose(result, expected, atol=1e-4), (
        f"Expected {expected}, got {result}"
    )


@pytest.mark.parametrize(
    ("predictions", "targets", "expected"),
    [
        (
            torch.tensor(
                [[[1.0, 0.0], [1.0, 0.0]], [[0.0, 1.0], [0.0, 1.0]]]
            ).unsqueeze(0),
            torch.tensor(
                [[[1.0, 0.0], [1.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]
            ).unsqueeze(0),
            torch.tensor([1.0, 0.0]),
        ),
        (
            torch.tensor(
                [[[1.0, 0.0], [1.0, 0.0]], [[0.0, 1.0], [0.0, 1.0]]]
            ).unsqueeze(0),
            torch.tensor(
                [[[1.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 1.0]]]
            ).unsqueeze(0),
            torch.tensor([0.5, 0.5]),
        ),
        (
            torch.tensor(
                [
                    [[1.0, 1.0], [0.0, 1.0]],
                    [[1.0, 1.0], [1.0, 1.0]],
                ]
            ).unsqueeze(0),
            torch.tensor(
                [
                    [[0.0, 1.0], [0.0, 1.0]],
                    [[1.0, 1.0], [1.0, 1.0]],
                ]
            ).unsqueeze(0),
            torch.tensor([2 / 3, 0.25]),
        ),
    ],
)
def test_mean_iou_per_class(
    predictions: Tensor, targets: Tensor, expected: Tensor
):
    metric = MIoU(
        num_classes=2,
        include_background=True,
        per_class=True,
        input_format="one-hot",
        node=DummyNodeSegmentation(n_classes=2),
    )

    metric.update(predictions, targets)
    result = metric.compute()
    assert torch.allclose(result, expected, atol=1e-4), (
        f"Expected {expected}, got {result}"
    )


def test_batch_updates():
    batch1_pred = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.0, 1.0], [1.0, 0.0]],
        ]
    ).unsqueeze(0)

    batch1_target = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.0, 1.0], [1.0, 0.0]],
        ]
    ).unsqueeze(0)

    batch2_pred = torch.tensor(
        [
            [[0.0, 0.0], [0.0, 0.0]],
            [[1.0, 1.0], [1.0, 1.0]],
        ]
    ).unsqueeze(0)

    batch2_target = torch.tensor(
        [
            [[1.0, 1.0], [1.0, 1.0]],
            [[0.0, 0.0], [0.0, 0.0]],
        ]
    ).unsqueeze(0)

    dice_metric = DiceCoefficient(
        num_classes=2,
        include_background=True,
        average="micro",
        input_format="one-hot",
        node=DummyNodeSegmentation(n_classes=2),
    )

    dice_metric.update(batch1_pred, batch1_target)
    dice_metric.update(batch2_pred, batch2_target)
    dice_result = dice_metric.compute()

    assert torch.isclose(dice_result, torch.tensor(0.5), atol=1e-4)

    iou_metric = MIoU(
        num_classes=2,
        include_background=True,
        input_format="one-hot",
        node=DummyNodeSegmentation(n_classes=2),
    )

    iou_metric.update(batch1_pred, batch1_target)
    iou_metric.update(batch2_pred, batch2_target)
    iou_result = iou_metric.compute()

    assert torch.isclose(iou_result, torch.tensor(0.5), atol=1e-4)


@pytest.mark.parametrize(
    ("predictions", "targets", "expected_dice", "expected_iou"),
    [
        (
            torch.zeros(1, 2, 2, 2),
            torch.zeros(1, 2, 2, 2),
            torch.tensor(0.0),
            torch.tensor(0.0),
        ),
        (
            torch.tensor(
                [
                    [[1.0, 0.0], [0.0, 0.0]],
                    [[0.0, 0.0], [0.0, 0.0]],
                ]
            ).unsqueeze(0),
            torch.tensor(
                [
                    [[1.0, 0.0], [0.0, 0.0]],
                    [[0.0, 0.0], [0.0, 0.0]],
                ]
            ).unsqueeze(0),
            torch.tensor(0.4),
            torch.tensor(0.125),
        ),
        (
            torch.tensor(
                [
                    [[1.0, 1.0], [1.0, 1.0]],
                    [[0.0, 0.0], [0.0, 0.0]],
                ]
            ).unsqueeze(0),
            torch.tensor(
                [
                    [[1.0, 1.0], [1.0, 1.0]],
                    [[0.0, 0.0], [0.0, 0.0]],
                ]
            ).unsqueeze(0),
            torch.tensor(1.0),
            torch.tensor(0.5),
        ),
    ],
)
def test_edge_cases(
    predictions: Tensor,
    targets: Tensor,
    expected_dice: Tensor,
    expected_iou: Tensor,
):
    dice_metric = DiceCoefficient(
        num_classes=2,
        include_background=True,
        average="micro",
        input_format="one-hot",
        node=DummyNodeSegmentation(n_classes=2),
    )

    dice_metric.update(predictions, targets)
    dice_result = dice_metric.compute()
    assert torch.isclose(dice_result, expected_dice, atol=1e-4)

    iou_metric = MIoU(
        num_classes=2,
        include_background=True,
        input_format="one-hot",
        node=DummyNodeSegmentation(n_classes=2),
    )

    iou_metric.update(predictions, targets)
    iou_result = iou_metric.compute()
    assert torch.isclose(iou_result, expected_iou, atol=1e-4)
